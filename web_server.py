# Web UI server for the local AI agent.
#
# Exposes the same capabilities as the CLI (chat, web research, book Q&A,
# file correction/summarization, RAG indexing, model + multi-agent
# management) over a small FastAPI app, with Server-Sent Events (SSE) for
# live phase updates during long-running operations.
#
# Run locally:
#     python web_server.py
# Or via Docker:
#     docker compose up
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

import agent as agent_module
import local_ollama
import rag


# --- shared app state -------------------------------------------------------


@dataclass
class AppState:
    config: local_ollama.OllamaConfig
    quality_mode: bool = True
    chat_memory: list[dict[str, str]] = field(default_factory=list)
    active_book: str | None = None
    active_index: rag.RAGIndex | None = None
    rag_dir: str = "./rag"
    out_dir: str = "./files"
    upload_dir: str = "./files"
    max_results: int = 10
    max_sources: int = 3
    max_chars: int = 6000
    rag_top_k: int = 5
    rag_max_chars_per_chunk: int = 1500
    rag_chunk_chars: int = 1200
    rag_overlap_chars: int = 200
    rag_min_chunk_chars: int = 200
    rag_max_chars: int = 2_000_000
    use_mcp: bool = False


def _build_state() -> AppState:
    cfg = local_ollama.OllamaConfig.from_env()
    rag_dir = os.getenv("AGENT_RAG_DIR", "./rag")
    out_dir = os.getenv("AGENT_OUT_DIR", "./files")
    Path(rag_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    state = AppState(config=cfg, rag_dir=rag_dir, out_dir=out_dir, upload_dir=out_dir)

    # Auto-load book on startup, like the CLI does.
    if agent_module._env_flag("AGENT_AUTO_BOOK", default=True):
        forced = (os.getenv("AGENT_DEFAULT_BOOK", "") or "").strip()
        try:
            available = rag.list_indexes(rag_dir=rag_dir)
        except Exception:
            available = []
        target = forced if forced and forced in available else (available[0] if available else "")
        if target:
            try:
                idx = rag.RAGIndex.load(rag.index_path(target, rag_dir=rag_dir))
                state.active_book = target
                state.active_index = idx
            except Exception:
                pass
    return state


STATE = _build_state()


# --- utilities --------------------------------------------------------------


def _trim_memory(memory: list[dict[str, str]]) -> list[dict[str, str]]:
    max_turns = max(0, agent_module._env_int("AGENT_MEMORY_TURNS", default=8))
    max_chars = max(200, agent_module._env_int("AGENT_MEMORY_MAX_CHARS", default=2000))
    if max_turns <= 0:
        return []
    # Walk backwards so a leading summary message (compressed earlier turns) is
    # always retained even when it pushes us slightly past the turn cap.
    cap = max_turns * 2
    tail = memory[-cap:] if len(memory) > cap else memory[:]
    if memory and agent_module._is_memory_summary(memory[0]) and (not tail or tail[0] is not memory[0]):
        tail = [memory[0], *tail]
    trimmed: list[dict[str, str]] = []
    for msg in tail:
        text = str(msg.get("content", "") or "")
        if len(text) > max_chars and not agent_module._is_memory_summary(msg):
            text = text[:max_chars].rstrip() + "...[truncated]"
        trimmed.append({"role": msg.get("role", "user"), "content": text})
    return trimmed


def _remember(
    state: AppState,
    user_text: str,
    assistant_text: str,
    *,
    on_status: Callable[[str], None] | None = None,
) -> None:
    state.chat_memory.append({"role": "user", "content": user_text})
    state.chat_memory.append({"role": "assistant", "content": assistant_text})
    state.chat_memory = _trim_memory(state.chat_memory)
    # Compression is intentionally NOT run here — it would block the response
    # path on a 200-400 token LLM call. It runs at the start of the *next*
    # turn instead (see _maybe_compress_state_memory call in _dispatch_message),
    # which folds the cost into the natural "Understanding request..." phase.


def _maybe_compress_state_memory(
    state: AppState,
    *,
    on_status: Callable[[str], None] | None = None,
) -> None:
    state.chat_memory = agent_module._maybe_compress_session_memory(
        state.chat_memory,
        config=state.config,
        on_status=on_status,
    )


def _ollama_get(state: AppState, path: str) -> dict[str, Any]:
    url = f"{state.config.host}{path}"
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=state.config.timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="replace"))


def _ollama_list_models(state: AppState) -> list[dict[str, Any]]:
    try:
        obj = _ollama_get(state, "/api/tags")
    except Exception:
        return []
    items = obj.get("models") or []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "name": str(item.get("name", "") or ""),
                "size_b": item.get("size"),
                "modified": item.get("modified_at"),
                "details": item.get("details") or {},
            }
        )
    out.sort(key=lambda r: r["name"].lower())
    return out


def _ollama_pull_stream(state: AppState, model: str) -> Any:
    url = f"{state.config.host}/api/pull"
    payload = json.dumps({"name": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
    )
    return urllib.request.urlopen(req, timeout=max(600.0, state.config.timeout_s))


# --- SSE helpers ------------------------------------------------------------


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _StatusBridge:
    """Bridges sync `on_status(text)` callbacks to an asyncio.Queue
    consumed by the SSE response generator."""

    SENTINEL: object = object()

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    def callback(self) -> Callable[[str], None]:
        def _cb(text: str) -> None:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, ("status", text))
            except Exception:
                pass

        return _cb

    async def events(self) -> AsyncIterator[tuple[str, Any]]:
        while True:
            item = await self._queue.get()
            if item is self.SENTINEL:
                return
            yield item

    def push(self, event: str, payload: Any) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (event, payload))

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, self.SENTINEL)


async def _run_with_status(
    work: Callable[..., dict[str, Any]],
    *,
    with_tokens: bool = False,
) -> AsyncIterator[str]:
    """Run a sync `work(on_status[, on_token])` function in a thread,
    streaming incremental status (and optionally token) events plus a final
    `result` event over SSE."""
    loop = asyncio.get_running_loop()
    bridge = _StatusBridge(loop)
    cb_status = bridge.callback()

    def cb_token(text: str) -> None:
        try:
            loop.call_soon_threadsafe(bridge._queue.put_nowait, ("token", {"text": text}))
        except Exception:
            pass

    def _runner() -> None:
        try:
            result = work(cb_status, cb_token) if with_tokens else work(cb_status)
            bridge.push("result", result)
        except Exception as exc:
            bridge.push("error", {"message": str(exc)})
        finally:
            bridge.close()

    threading.Thread(target=_runner, daemon=True).start()

    yield _sse("ready", {"ts": time.time()})
    last_status = ""
    async for event, payload in bridge.events():
        if event == "status":
            text = str(payload or "").strip()
            if not text or text == last_status:
                continue
            last_status = text
            yield _sse("status", {"text": text})
        else:
            yield _sse(event, payload)


# --- core dispatch (mirrors agent.py chat loop) -----------------------------


def _dispatch_message(
    state: AppState,
    message: str,
    on_status: Callable[[str], None],
    on_token: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Mirror of agent.py's chat-loop dispatch, returning a structured
    result instead of printing. Used by `/api/chat`.

    When `on_token` is provided AND the chat path is single-model (no
    committee), assistant content is streamed chunk-by-chunk via that
    callback so the UI can render tokens live."""
    config = state.config
    has_active_book = state.active_index is not None and state.active_book is not None

    # Run any pending memory compression at the *start* of this turn so the
    # cost is folded into the visible "Understanding request..." phase rather
    # than blocking the previous response.
    _maybe_compress_state_memory(state, on_status=on_status)

    # Sub-agent decomposition runs before single-mode routing. When the planner
    # decides the request is multi-part it spawns parallel sub-agents (chat or
    # research scopes) and synthesizes a final answer; otherwise we fall through
    # to the regular single-mode pipeline below.
    research_kwargs = {
        "max_results": state.max_results,
        "max_sources": state.max_sources,
        "max_chars_per_source": state.max_chars,
        "use_mcp": state.use_mcp,
        "verify_answers": state.quality_mode,
    }
    sub = agent_module._maybe_dispatch_subagents(
        message=message,
        config=config,
        on_status=on_status,
        research_kwargs=research_kwargs,
    )
    if sub is not None:
        answer = (sub.get("answer") or "").rstrip()
        sources = list(sub.get("sources") or [])
        _remember(state, message, answer, on_status=on_status)
        return {
            "mode": "subagents",
            "answer": answer,
            "sources": sources,
            "active_book": state.active_book,
            "subagents": [
                {"scope": p["scope"], "prompt": p["prompt"]}
                for p in (sub.get("plan") or [])
            ],
        }

    on_status("Understanding request...")
    mode, route = agent_module._decide_mode_for_prompt(
        message,
        has_active_book=has_active_book,
        config=config,
        on_status=on_status,
    )

    sources: list[str] = []
    answer = ""
    final_mode = mode

    if mode == "book" and state.active_index is not None and state.active_book is not None:
        on_status(f"[Mode] book - retrieving passages from '{state.active_book}'")
        ans, srcs = agent_module.book_answer(
            message,
            index=state.active_index,
            book_name=state.active_book,
            config=config,
            top_k=state.rag_top_k,
            max_chars_per_chunk=state.rag_max_chars_per_chunk,
            verify_answers=state.quality_mode,
            on_status=on_status,
        )
        answer = ans
        sources = list(srcs or [])
    else:
        action = "research" if mode == "web" else mode

        if action == "research":
            urls_in_msg = agent_module._extract_urls(message)
            query = (route.get("query") or "").strip() or message
            on_status("[Mode] web - searching web + reading sources")
            # `research_answer` is async but this dispatcher runs in a worker
            # thread (no event loop). asyncio.run creates one for the thread.
            ans, urls = asyncio.run(
                agent_module.research_answer(
                    query,
                    config=config,
                    max_results=state.max_results,
                    max_sources=state.max_sources,
                    max_chars_per_source=state.max_chars,
                    use_mcp=state.use_mcp,
                    verify_answers=state.quality_mode,
                    seed_urls=urls_in_msg or None,
                    on_status=on_status,
                )
            )
            answer = ans
            sources = list(urls or [])
            final_mode = "web"

        elif action == "correct":
            target = (route.get("text") or "").strip() or message
            target = agent_module._strip_leading_instruction(target, action="correct")
            on_status("[Mode] correct - proofreading text")
            answer = agent_module.correct_text_resilient(
                target,
                language=route.get("language", "auto"),
                config=config,
            )
            final_mode = "correct"

        elif action == "summarize":
            target = (route.get("text") or "").strip() or message
            target = agent_module._strip_leading_instruction(target, action="summarize")
            on_status("[Mode] summarize - summarizing text")
            answer = agent_module.summarize_text(
                target,
                language=route.get("language", "auto"),
                length=route.get("length", "short"),
                config=config,
            )
            final_mode = "summarize"

        else:  # chat
            on_status("[Mode] chat - general answer")
            prompt = (route.get("text") or "").strip() or message
            messages = [
                {"role": "system", "content": agent_module._CHAT_SYSTEM_PROMPT},
                *_trim_memory(state.chat_memory),
                {"role": "user", "content": prompt},
            ]
            chat_options = {"temperature": 0.2, "num_ctx": 4096, "num_predict": 800}

            # Stream tokens only when single-model — committee mode has its
            # own multi-stage pipeline (drafting/review/merge) that can't
            # stream meaningfully.
            committee_on = agent_module._committee_enabled_for("chat")
            if on_token is not None and not committee_on:
                on_status("Thinking...")
                buf: list[str] = []
                for chunk in local_ollama.chat_stream(
                    config=config, messages=messages, options=chat_options
                ):
                    buf.append(chunk)
                    on_token(chunk)
                answer = "".join(buf)
            else:
                answer = agent_module._generate_answer(
                    scope="chat",
                    config=config,
                    messages=messages,
                    options=chat_options,
                    on_status=on_status,
                    status_label="Thinking...",
                )
            final_mode = "chat"

    answer = (answer or "").rstrip()
    _remember(state, message, answer, on_status=on_status)
    return {
        "mode": final_mode,
        "answer": answer,
        "sources": sources,
        "active_book": state.active_book,
    }


# --- FastAPI app ------------------------------------------------------------


app = FastAPI(title="Local AI Agent")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Web UI not built — static/index.html missing.</h1>", status_code=500)
    return HTMLResponse(index.read_text(encoding="utf-8"))


def _whisperx_reachable() -> bool:
    # Quick liveness probe — the whisperx service is opt-in (compose profile
    # "voice"). When it's not running we want the web UI to gracefully hide
    # the mic button rather than show a broken control.
    url = f"{_whisperx_url()}/healthz"
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


@app.get("/api/health")
async def health() -> dict[str, Any]:
    ok = True
    err = ""
    try:
        _ollama_get(STATE, "/api/tags")
    except Exception as exc:
        ok = False
        err = str(exc)
    # Capability probe runs in a thread so we don't block the event loop on
    # whichever DNS/TCP timeout the platform applies when whisperx is absent.
    transcription = await asyncio.get_running_loop().run_in_executor(None, _whisperx_reachable)
    return {
        "ok": ok,
        "ollama_host": STATE.config.host,
        "primary_model": STATE.config.model,
        "error": err,
        "capabilities": {
            "transcription": transcription,
        },
    }


# --- chat (SSE) -------------------------------------------------------------


@app.post("/api/chat")
async def chat_endpoint(payload: dict[str, Any]) -> StreamingResponse:
    message = str(payload.get("message", "") or "").strip()
    if not message:
        raise HTTPException(400, "Empty message.")

    def _work(on_status: Callable[[str], None], on_token: Callable[[str], None]) -> dict[str, Any]:
        return _dispatch_message(STATE, message, on_status, on_token=on_token)

    return StreamingResponse(_run_with_status(_work, with_tokens=True), media_type="text/event-stream")


# --- explicit slash-command equivalents -------------------------------------


@app.post("/api/research")
async def research_endpoint(payload: dict[str, Any]) -> StreamingResponse:
    query = str(payload.get("query", "") or "").strip()
    if not query:
        raise HTTPException(400, "Empty query.")

    def _work(on_status: Callable[[str], None]) -> dict[str, Any]:
        on_status("[Mode] web - searching web + reading sources")
        ans, urls = asyncio.run(
            agent_module.research_answer(
                query,
                config=STATE.config,
                max_results=STATE.max_results,
                max_sources=STATE.max_sources,
                max_chars_per_source=STATE.max_chars,
                use_mcp=STATE.use_mcp,
                verify_answers=STATE.quality_mode,
                on_status=on_status,
            )
        )
        answer = (ans or "").rstrip()
        _remember(STATE, f"/research {query}", answer)
        return {"mode": "web", "answer": answer, "sources": list(urls or [])}

    return StreamingResponse(_run_with_status(_work), media_type="text/event-stream")


# --- book / RAG -------------------------------------------------------------


@app.get("/api/books")
async def list_books() -> dict[str, Any]:
    try:
        items = rag.list_indexes(rag_dir=STATE.rag_dir)
    except Exception as exc:
        raise HTTPException(500, f"Failed to list books: {exc}")
    return {
        "books": list(items),
        "active": STATE.active_book,
    }


@app.post("/api/books/load")
async def load_book(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name", "") or "").strip()
    if not name:
        raise HTTPException(400, "Missing book name.")
    try:
        idx = rag.RAGIndex.load(rag.index_path(name, rag_dir=STATE.rag_dir))
    except Exception as exc:
        raise HTTPException(404, f"Failed to load book '{name}': {exc}")
    STATE.active_book = name
    STATE.active_index = idx
    return {"active": STATE.active_book, "chunks": len(idx.chunks)}


@app.post("/api/books/off")
async def unload_book() -> dict[str, Any]:
    STATE.active_book = None
    STATE.active_index = None
    return {"active": None}


@app.post("/api/books/index")
async def index_book(file: UploadFile = File(...), name: str = Form("")) -> StreamingResponse:
    upload_dir = Path(STATE.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload").name
    dest = agent_module._unique_output_path(upload_dir / safe_name)
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    book_name = rag.sanitize_index_name((name or "").strip() or dest.stem)

    def _work(on_status: Callable[[str], None]) -> dict[str, Any]:
        on_status("Reading file...")
        text, page_spans = agent_module._read_for_rag_index(str(dest), max_chars=STATE.rag_max_chars)
        if not text:
            raise RuntimeError("No text could be extracted from the file.")
        on_status("Building index...")
        idx = rag.build_index_from_text(
            text,
            source={"file": str(dest)},
            page_spans=page_spans,
            chunk_chars=STATE.rag_chunk_chars,
            overlap_chars=STATE.rag_overlap_chars,
            min_chunk_chars=STATE.rag_min_chunk_chars,
        )
        on_status("Saving index...")
        out_path = rag.index_path(book_name, rag_dir=STATE.rag_dir)
        idx.save(out_path)
        STATE.active_book = book_name
        STATE.active_index = idx
        return {
            "name": book_name,
            "chunks": len(idx.chunks),
            "saved_file": str(dest),
            "index_file": str(out_path),
        }

    return StreamingResponse(_run_with_status(_work), media_type="text/event-stream")


# --- file transforms (correct / summarize) ----------------------------------


def _save_upload(file: UploadFile) -> Path:
    upload_dir = Path(STATE.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload").name
    dest = agent_module._unique_output_path(upload_dir / safe_name)
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return dest


@app.post("/api/correct")
async def correct_endpoint(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    language: str = Form("auto"),
) -> StreamingResponse:
    if file is None and not text.strip():
        raise HTTPException(400, "Provide either text or a file.")
    saved_path = _save_upload(file) if file is not None else None

    def _work(on_status: Callable[[str], None]) -> dict[str, Any]:
        if saved_path is not None:
            out_path = agent_module._unique_output_path(
                agent_module._safe_output_path(
                    out_dir=STATE.out_dir,
                    requested_path="",
                    input_path=str(saved_path),
                    action="correct",
                )
            )
            agent_module._save_file_transform(
                input_path=saved_path,
                output_path=out_path,
                action="correct",
                config=STATE.config,
                language=language,
                ocr_language=agent_module._ocr_language_from_prompt("", preferred_language=language),
                on_status=on_status,
            )
            return {
                "saved_path": str(out_path),
                "download_name": out_path.name,
            }
        on_status("Correcting...")
        out = agent_module.correct_text_resilient(text, language=language, config=STATE.config)
        return {"text": out}

    return StreamingResponse(_run_with_status(_work), media_type="text/event-stream")


@app.post("/api/summarize")
async def summarize_endpoint(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    language: str = Form("auto"),
    length: str = Form("short"),
) -> StreamingResponse:
    if file is None and not text.strip():
        raise HTTPException(400, "Provide either text or a file.")
    saved_path = _save_upload(file) if file is not None else None

    def _work(on_status: Callable[[str], None]) -> dict[str, Any]:
        body = text
        if saved_path is not None:
            on_status("Reading file...")
            file_text, err = agent_module._maybe_read_local_file(
                str(saved_path),
                ocr_language=agent_module._ocr_language_from_prompt("", preferred_language=language),
                on_status=on_status,
            )
            if err:
                raise RuntimeError(err)
            body = file_text or ""
        if not body.strip():
            raise RuntimeError("No text to summarize.")
        on_status("Summarizing...")
        out = agent_module.summarize_text(body, language=language, length=length, config=STATE.config)
        return {"text": out, "saved_path": str(saved_path) if saved_path else None}

    return StreamingResponse(_run_with_status(_work), media_type="text/event-stream")


# --- voice transcription (WhisperX) ----------------------------------------


def _whisperx_url() -> str:
    return (os.getenv("WHISPERX_URL", "http://whisperx:9000") or "").rstrip("/")


def _whisperx_timeout() -> float:
    try:
        return max(5.0, float(os.getenv("WHISPERX_TIMEOUT_S", "120")))
    except Exception:
        return 120.0


_FILENAME_UNSAFE_RE = re.compile(r'[\r\n"\\]+')


def _sanitize_upload_filename(name: str, *, fallback: str) -> str:
    # Strip CR/LF (header injection risk) and quotes (would break the
    # Content-Disposition wrapping). Cap length so a pathological filename
    # can't blow up the request line.
    cleaned = _FILENAME_UNSAFE_RE.sub("", (name or "").strip())
    cleaned = cleaned[:120].strip()
    return cleaned or fallback


def _multipart_body(filename: str, content: bytes, ctype: str, language: str) -> tuple[bytes, str]:
    boundary = "----audioupload" + os.urandom(8).hex()
    safe_name = _sanitize_upload_filename(filename, fallback="voice.webm")
    parts: list[bytes] = []
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n".encode("utf-8")
    )
    parts.append(content)
    parts.append(b"\r\n")
    if language:
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(b'Content-Disposition: form-data; name="language"\r\n\r\n')
        parts.append(language.encode("utf-8"))
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _max_audio_bytes() -> int:
    # Default 25 MB — generous for short voice messages, prevents arbitrary
    # large blobs from tying up the WhisperX worker.
    try:
        return max(64_000, int(os.getenv("WHISPERX_MAX_UPLOAD_BYTES", "25000000")))
    except Exception:
        return 25_000_000


@app.post("/api/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str = Form(""),
) -> dict[str, Any]:
    # Voice is opt-in (compose profile "voice"). When the WhisperX service
    # isn't running we return a calm, intentional 503 — no leaked exception
    # text, no exception backtrace in the access log — so an operator who
    # never opted into voice doesn't see anything that looks like a bug.
    loop = asyncio.get_running_loop()
    if not await loop.run_in_executor(None, _whisperx_reachable):
        raise HTTPException(503, "Voice transcription is disabled.")

    max_bytes = _max_audio_bytes()
    chunks: list[bytes] = []
    total = 0
    # Stream the upload so we can reject oversize blobs without ever holding
    # them fully in memory. UploadFile.read(N) reads up to N bytes per call.
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, f"Audio upload exceeds {max_bytes} bytes.")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(400, "Empty audio upload.")
    audio = b"".join(chunks)

    body, ctype = _multipart_body(
        filename=file.filename or "voice.webm",
        content=audio,
        ctype=file.content_type or "audio/webm",
        language=(language or "").strip(),
    )

    url = f"{_whisperx_url()}/transcribe"
    req = urllib.request.Request(
        url,
        method="POST",
        data=body,
        headers={"Content-Type": ctype, "Accept": "application/json"},
    )

    def _post() -> dict[str, Any]:
        with urllib.request.urlopen(req, timeout=_whisperx_timeout()) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8", errors="replace"))

    try:
        data = await loop.run_in_executor(None, _post)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise HTTPException(exc.code or 502, f"WhisperX error: {detail}")
    except Exception:
        # Race: whisperx was reachable at the precheck above but went away by
        # the time we tried to proxy. Same calm response as the opt-out case.
        raise HTTPException(503, "Voice transcription is disabled.")

    text = str(data.get("text", "") or "").strip()
    return {
        "text": text,
        "language": str(data.get("language", "") or ""),
    }


@app.get("/api/files/{name}")
async def download_file(name: str) -> FileResponse:
    safe = Path(name).name
    target = Path(STATE.out_dir) / safe
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found.")
    return FileResponse(str(target), filename=safe)


# --- settings (memory / quality / models / committee) ----------------------


@app.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    return {
        "primary_model": STATE.config.model,
        "host": STATE.config.host,
        "quality_mode": STATE.quality_mode,
        "multi_agent": agent_module._env_flag("AGENT_MULTI_AGENT", default=False),
        "multi_models": [m.strip() for m in (os.getenv("AGENT_MULTI_MODELS", "") or "").split(",") if m.strip()],
        "multi_scopes": [s.strip() for s in (os.getenv("AGENT_MULTI_SCOPES", "chat,research,book,summarize,correct") or "").split(",") if s.strip()],
        "multi_smart": agent_module._env_flag("AGENT_MULTI_SMART", default=True),
        "multi_role_mode": agent_module._env_flag("AGENT_MULTI_ROLE_MODE", default=True),
        "multi_peer_review": agent_module._env_flag("AGENT_MULTI_PEER_REVIEW", default=True),
        "multi_simple_models": agent_module._env_int("AGENT_MULTI_SIMPLE_MODELS", default=1),
        "multi_medium_models": agent_module._env_int("AGENT_MULTI_MEDIUM_MODELS", default=2),
        "multi_hard_models": agent_module._env_int("AGENT_MULTI_HARD_MODELS", default=0),
        "memory_turns": len(STATE.chat_memory) // 2,
        "max_memory_turns": agent_module._env_int("AGENT_MEMORY_TURNS", default=8),
        "memory_compress": agent_module._env_flag("AGENT_MEMORY_COMPRESS", default=True),
        "memory_compress_threshold": agent_module._env_int("AGENT_MEMORY_COMPRESS_THRESHOLD_CHARS", default=8000),
        "memory_keep_recent_turns": agent_module._env_int("AGENT_MEMORY_KEEP_RECENT_TURNS", default=4),
        "subagents": agent_module._env_flag("AGENT_SUBAGENTS", default=False),
        "subagents_max": agent_module._env_int("AGENT_SUBAGENTS_MAX", default=3),
    }


_BOOL_TO_ENV = lambda v: "on" if bool(v) else "off"


@app.post("/api/settings")
async def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    if "quality_mode" in payload:
        STATE.quality_mode = bool(payload["quality_mode"])

    # Multi-agent settings live in os.environ because the agent module reads
    # them via `_env_flag` / `_env_int` on every committee call — mutating
    # the env makes UI changes take effect immediately, no restart.
    if "primary_model" in payload:
        new_model = str(payload["primary_model"] or "").strip()
        if new_model:
            STATE.config = local_ollama.OllamaConfig(
                host=STATE.config.host,
                model=new_model,
                timeout_s=STATE.config.timeout_s,
            )
            os.environ["OLLAMA_MODEL"] = new_model

    if "multi_agent" in payload:
        os.environ["AGENT_MULTI_AGENT"] = _BOOL_TO_ENV(payload["multi_agent"])

    if "multi_models" in payload:
        models = payload["multi_models"]
        if isinstance(models, list):
            os.environ["AGENT_MULTI_MODELS"] = ",".join(str(m).strip() for m in models if str(m).strip())
        elif isinstance(models, str):
            os.environ["AGENT_MULTI_MODELS"] = models.strip()

    if "multi_scopes" in payload:
        scopes = payload["multi_scopes"]
        if isinstance(scopes, list):
            os.environ["AGENT_MULTI_SCOPES"] = ",".join(str(s).strip() for s in scopes if str(s).strip())

    for key, env_name in (
        ("multi_smart", "AGENT_MULTI_SMART"),
        ("multi_role_mode", "AGENT_MULTI_ROLE_MODE"),
        ("multi_peer_review", "AGENT_MULTI_PEER_REVIEW"),
        ("memory_compress", "AGENT_MEMORY_COMPRESS"),
        ("subagents", "AGENT_SUBAGENTS"),
    ):
        if key in payload:
            os.environ[env_name] = _BOOL_TO_ENV(payload[key])

    for key, env_name in (
        ("multi_simple_models", "AGENT_MULTI_SIMPLE_MODELS"),
        ("multi_medium_models", "AGENT_MULTI_MEDIUM_MODELS"),
        ("multi_hard_models", "AGENT_MULTI_HARD_MODELS"),
        ("memory_compress_threshold", "AGENT_MEMORY_COMPRESS_THRESHOLD_CHARS"),
        ("memory_keep_recent_turns", "AGENT_MEMORY_KEEP_RECENT_TURNS"),
        ("subagents_max", "AGENT_SUBAGENTS_MAX"),
    ):
        if key in payload:
            try:
                os.environ[env_name] = str(int(payload[key]))
            except Exception:
                pass

    return await get_settings()


# --- model management ------------------------------------------------------


@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    return {
        "installed": _ollama_list_models(STATE),
        "primary": STATE.config.model,
        "committee": [m.strip() for m in (os.getenv("AGENT_MULTI_MODELS", "") or "").split(",") if m.strip()],
    }


@app.post("/api/models/pull")
async def pull_model(payload: dict[str, Any]) -> StreamingResponse:
    name = str(payload.get("name", "") or "").strip()
    if not name:
        raise HTTPException(400, "Missing model name.")

    async def _gen() -> AsyncIterator[str]:
        yield _sse("ready", {"model": name})
        loop = asyncio.get_running_loop()
        q: queue.Queue[Any] = queue.Queue()

        def _worker() -> None:
            try:
                resp = _ollama_pull_stream(STATE, name)
                for raw in resp:
                    if not raw:
                        continue
                    try:
                        line = raw.decode("utf-8", errors="replace").strip()
                    except Exception:
                        continue
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    q.put(("status", obj))
                    if obj.get("status") == "success":
                        break
                q.put(("done", {"model": name}))
            except urllib.error.HTTPError as exc:
                q.put(("error", {"message": f"HTTP {exc.code}: {exc.reason}"}))
            except Exception as exc:
                q.put(("error", {"message": str(exc)}))
            finally:
                q.put(None)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                return
            event, payload = item
            yield _sse(event, payload)

    return StreamingResponse(_gen(), media_type="text/event-stream")


# --- memory -----------------------------------------------------------------


@app.get("/api/memory")
async def get_memory() -> dict[str, Any]:
    return {
        "turns": len(STATE.chat_memory) // 2,
        "max_turns": agent_module._env_int("AGENT_MEMORY_TURNS", default=8),
        "messages": STATE.chat_memory,
    }


@app.post("/api/memory/clear")
async def clear_memory() -> dict[str, Any]:
    STATE.chat_memory = []
    return {"turns": 0}


# --- entrypoint -------------------------------------------------------------


def main() -> None:
    import uvicorn

    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8000"))
    uvicorn.run("web_server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
