from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path


_STDLIB_TOPLEVEL = {
    "argparse",
    "asyncio",
    "codecs",
    "collections",
    "contextlib",
    "dataclasses",
    "datetime",
    "html",
    "io",
    "json",
    "math",
    "mimetypes",
    "os",
    "pathlib",
    "re",
    "secrets",
    "shlex",
    "sys",
    "typing",
    "urllib",
}

_LOCAL_TOPLEVEL = {
    "agent",
    "file_readers",
    "file_writers",
    "local_ollama",
    "mcp_web_tools",
    "ocr_api",
    "rag",
    "web_tools",
}


def _toplevel_module(mod: str) -> str:
    mod = (mod or "").strip()
    return mod.split(".", 1)[0] if mod else ""


def _classify_module(mod: str) -> str:
    top = _toplevel_module(mod)
    if top in _LOCAL_TOPLEVEL:
        return "local"
    if top in _STDLIB_TOPLEVEL:
        return "stdlib"
    return "third-party"


def _explain_import(line: str) -> str | None:
    s = line.strip()
    if s.startswith("import "):
        mods = [m.strip() for m in s[len("import ") :].split(",") if m.strip()]
        if not mods:
            return None
        kinds = {_classify_module(m.split(" as ", 1)[0].strip()) for m in mods}
        kind = "imports"
        if kinds == {"stdlib"}:
            kind = "Imports standard library modules"
        elif kinds == {"local"}:
            kind = "Imports local project modules"
        elif "third-party" in kinds:
            kind = "Imports third-party modules"
        else:
            kind = "Imports modules"
        return f"{kind}: {', '.join(mods)}."

    if s.startswith("from ") and " import " in s:
        mod = s.split(" import ", 1)[0].replace("from ", "").strip()
        what = s.split(" import ", 1)[1].strip()
        kind = _classify_module(mod)
        if kind == "stdlib":
            return f"Imports {what} from the standard library module `{mod}`."
        if kind == "local":
            return f"Imports {what} from the local module `{mod}`."
        return f"Imports {what} from the third-party module `{mod}`."

    return None


_DEF_RE = re.compile(r"^(?P<async>async\s+)?def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CLASS_RE = re.compile(r"^class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")


_FILE_OVERVIEW: dict[str, str] = {
    "agent.py": (
        "Main CLI/chat entrypoint. Routes user requests to: proofread (correct), summarize, "
        "web research (search + fetch + cited answer), or normal chat via a local Ollama model."
    ),
    "rag.py": "Simple local RAG: chunk text, build BM25 index, and retrieve relevant passages for Q&A.",
    "web_tools.py": (
        "HTTPS-only web utilities: search (DuckDuckGo HTML/Lite/Instant Answer) and fetch+extract text "
        "(HTML/PDF/DOCX/PPTX/XLSX/images via OCR.Space)."
    ),
    "file_readers.py": "Local file text extraction (PDF/DOCX/PPTX/XLSX/HTML/text + image OCR via OCR.Space).",
    "file_writers.py": "Safe local output writing helpers (writes text or DOCX to the output directory).",
    "local_ollama.py": "Minimal Ollama HTTP client + Gemma3 1B/4B model validation.",
    "ocr_api.py": "OCR.Space HTTPS client for doing OCR on images.",
    "mcp_web_tools.py": "FastMCP server exposing `web_search` and `fetch_url` tools over stdio.",
    "compose.yaml": "Docker Compose stack (agent + Ollama) with hardening and no published ports.",
    "Dockerfile": "Builds the agent container image (Python slim + deps, run as non-root).",
    "requirements.txt": "Python dependencies for the agent.",
}


_DEF_HINTS: dict[str, dict[str, str]] = {
    "agent.py": {
        "_read_text": "Read input text from `--text`, `--file`, or stdin.",
        "_maybe_read_local_file": "Try to interpret a user message as a local file path and read it.",
        "_out_dir": "Resolve the output directory for generated files (env `AGENT_OUT_DIR` or `./files`).",
        "_build_config": "Build validated Ollama config (host/model/timeout) from args + env.",
        "correct_text": "Proofread text (spelling/grammar/punctuation) without rewriting.",
        "summarize_text": "Summarize the given text at a requested length.",
        "_extract_urls": "Extract URLs from free-form text.",
        "_simple_intent": "Heuristic intent detection (correct/summarize/research).",
        "_strip_leading_instruction": "Remove leading 'Correct this:'/'Summarize:' phrasing.",
        "_route_with_llm": "Ask the local LLM to produce a routing JSON decision.",
        "research_answer": "Search + fetch pages, then ask LLM to answer with citations [1], [2], ...",
        "cmd_correct": "CLI handler for `correct`.",
        "cmd_summarize": "CLI handler for `summarize`.",
        "cmd_research": "CLI handler for `research`.",
        "cmd_chat": "CLI handler for interactive chat loop.",
        "build_parser": "Define argparse CLI structure.",
        "main": "Async main that dispatches to the chosen subcommand.",
    },
    "rag.py": {
        "default_rag_dir": "Get the default directory for indexes (env `AGENT_RAG_DIR` or `./rag`).",
        "sanitize_index_name": "Make a safe filename-friendly index name.",
        "index_path": "Compute the path to an index file for a given name.",
        "list_indexes": "List available book indexes in the index directory.",
        "build_index_from_text": "Chunk a document and build a BM25 index for retrieval.",
    },
    "web_tools.py": {
        "ensure_https_url": "Normalize/upgrade URLs so only `https://` is allowed.",
        "fetch_url": "Fetch a URL and extract readable text based on content type/extension.",
        "web_search": "DuckDuckGo search with multiple fallbacks; returns HTTPS-only result URLs.",
    },
    "file_readers.py": {
        "read_any_file": "Best-effort text extraction for many local file types.",
    },
    "file_writers.py": {
        "write_text_file": "Write UTF-8 text to a local file (creating parent directories).",
        "write_docx_file": "Write text into a new DOCX (one paragraph per line).",
        "write_any_file": "Choose an output writer based on the target extension.",
    },
    "local_ollama.py": {
        "validate_gemma3_model": "Allow only `gemma3:1b` or `gemma3:4b` (with optional suffix).",
        "_normalize_host": "Normalize Ollama host string (scheme + no trailing slash).",
        "chat": "Call Ollama's `/api/chat` endpoint (non-streaming).",
    },
    "ocr_api.py": {
        "ocr_image_bytes": "Send image bytes to OCR.Space and return extracted text.",
    },
}


def _explain_line(path: Path, line: str) -> str:
    s = line.rstrip("\n")
    stripped = s.strip()

    if stripped == "":
        return "Blank line for readability."

    if stripped.startswith("#"):
        return "Comment/documentation line."

    imp = _explain_import(s)
    if imp:
        if s[: len(s) - len(s.lstrip())]:
            return "Lazy/inner-scope " + imp[0].lower() + imp[1:]
        return imp

    if stripped.startswith("@"):
        return "Decorator line: modifies the behavior of the next function/method."

    m = _CLASS_RE.match(stripped)
    if m:
        name = m.group("name")
        if name.endswith("Error") or name.endswith("Exception"):
            return f"Defines a custom exception class `{name}`."
        return f"Defines a class `{name}`."

    m = _DEF_RE.match(stripped)
    if m:
        name = m.group("name")
        hint = _DEF_HINTS.get(path.name, {}).get(name)
        if hint:
            return f"Defines `{name}()`: {hint}"
        return f"Defines function `{name}()`."

    head = stripped.split(" ", 1)[0]
    if head in {"if", "elif"}:
        return "Conditional branch: checks a condition and chooses a code path."
    if head == "else:" or stripped == "else:":
        return "Fallback branch for the preceding `if`/`elif`."
    if head in {"for", "while"}:
        return "Loop: repeats the following block."
    if head == "try:" or stripped == "try:":
        return "Start of a `try` block for exception handling."
    if head == "except":
        return "Exception handler: runs if the `try` block raises an error."
    if head == "finally:" or stripped == "finally:":
        return "Finally block: runs whether or not an error occurred."
    if head == "with":
        return "Context manager block: ensures setup/teardown around a resource."
    if head == "return":
        return "Returns a value from the current function."
    if head == "raise":
        return "Raises an exception to signal an error."
    if stripped in {"pass", "continue", "break"}:
        return "Control-flow keyword."

    if "=" in stripped and "==" not in stripped and "!=" not in stripped and not stripped.startswith(("return ", "raise ")):
        left = stripped.split("=", 1)[0].strip()
        if left:
            return f"Assignment: sets `{left}`."
        return "Assignment statement."

    if stripped.endswith(":"):
        return "Starts a new block (indented section) in Python."

    return "Implementation detail: part of the surrounding logic."


def _render_file(path: Path) -> str:
    src = path.read_text(encoding="utf-8", errors="replace").splitlines()
    now = _dt.datetime.now().isoformat(timespec="seconds")

    overview = _FILE_OVERVIEW.get(path.name, "")
    header = [f"# {path.name} — line-by-line explanation", "", f"Generated: {now}", ""]
    if overview:
        header.extend(["Purpose: " + overview, ""])

    header.extend(
        [
            "Format: each original source line is shown with its line number, followed by a short explanation.",
            "",
            "```text",
        ]
    )

    body: list[str] = []
    width = max(3, len(str(len(src))))
    for i, line in enumerate(src, start=1):
        body.append(f"{i:>{width}} {line}")
        body.append(f"{'':>{width}}   ↳ {_explain_line(path, line)}")

    footer = ["```", ""]
    return "\n".join(header + body + footer)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "docs" / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        repo / "agent.py",
        repo / "rag.py",
        repo / "web_tools.py",
        repo / "file_readers.py",
        repo / "file_writers.py",
        repo / "local_ollama.py",
        repo / "ocr_api.py",
        repo / "mcp_web_tools.py",
        repo / "Dockerfile",
        repo / "compose.yaml",
        repo / "requirements.txt",
    ]

    index_lines = [
        "# Annotated Code (line-by-line)",
        "",
        "These files contain line-by-line explanations of the source/config files in this repo.",
        "",
        "Regenerate anytime with:",
        "",
        "```powershell",
        "python tools/generate_annotated_docs.py",
        "```",
        "",
        "## Files",
    ]

    for path in targets:
        if not path.exists():
            continue
        rendered = _render_file(path)
        out_path = out_dir / f"{path.name}.md"
        out_path.write_text(rendered, encoding="utf-8", newline="\n")
        index_lines.append(f"- `docs/annotated/{path.name}.md`")

    (repo / "docs" / "ANNOTATED.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
