# Lightweight local RAG implementation.
# Handles chunking, BM25 indexing, index persistence, and retrieval.
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RAGError(RuntimeError):
    # Raised for index validation, IO, and retrieval failures.
    pass


# File extension used for persisted index payloads.
_INDEX_EXT = ".rag.json"
# Unicode-aware token matcher (letters only; excludes underscore).
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def default_rag_dir() -> Path:
    # Resolve RAG directory from env with project-local default.
    return Path(os.getenv("AGENT_RAG_DIR", "rag")).resolve()


def sanitize_index_name(name: str) -> str:
    # Normalize whitespace and reject empty names.
    name = (name or "").strip()
    if not name:
        raise RAGError("Index name is empty.")
    # Keep a filesystem-safe subset to avoid invalid filenames.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    if not cleaned:
        raise RAGError(f"Invalid index name: {name!r}")
    return cleaned


def index_path(name: str, *, rag_dir: str | Path | None = None) -> Path:
    # Resolve target index directory and ensure it exists.
    base = Path(rag_dir).resolve() if rag_dir is not None else default_rag_dir()
    base.mkdir(parents=True, exist_ok=True)
    # Build final `<safe_name>.rag.json` path.
    safe = sanitize_index_name(name)
    return base / f"{safe}{_INDEX_EXT}"


def list_indexes(*, rag_dir: str | Path | None = None) -> list[str]:
    # Resolve index directory and return empty list when absent.
    base = Path(rag_dir).resolve() if rag_dir is not None else default_rag_dir()
    if not base.exists():
        return []
    out: list[str] = []
    for p in sorted(base.glob(f"*{_INDEX_EXT}")):
        if p.is_file():
            out.append(p.name[: -len(_INDEX_EXT)])
    return out


def _tokenize(text: str) -> list[str]:
    # Lowercase tokens and drop very short terms to reduce BM25 noise.
    toks = [t.lower() for t in _WORD_RE.findall(text or "")]
    return [t for t in toks if len(t) >= 2]


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int, min_chunk_chars: int) -> list[tuple[int, int, str]]:
    # Validate chunking parameters early.
    if chunk_chars <= 0:
        raise RAGError("chunk_chars must be > 0")
    if overlap_chars < 0:
        raise RAGError("overlap_chars must be >= 0")
    if min_chunk_chars < 0:
        raise RAGError("min_chunk_chars must be >= 0")
    # Ensure overlap is always smaller than chunk size.
    overlap_chars = min(overlap_chars, max(0, chunk_chars - 1))

    # Normalize line endings and trim excessive whitespace blocks.
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)

    chunks: list[tuple[int, int, str]] = []
    i = 0
    n = len(s)
    while i < n:
        # Start with fixed-size window, then try semantic breakpoints.
        target_end = min(n, i + chunk_chars)
        end = target_end

        # Prefer split positions in the tail of the window (paragraph, line, sentence).
        window_start = i + int(chunk_chars * 0.6)
        window_start = min(window_start, target_end)
        candidates = [
            s.rfind("\n\n", window_start, target_end),
            s.rfind("\n", window_start, target_end),
            s.rfind(". ", window_start, target_end),
        ]
        best = max(candidates)
        if best != -1 and best > i:
            end = best + (2 if s.startswith(". ", best) else 0)
            end = max(end, i + 1)

        # Keep only chunks that satisfy minimum size.
        piece = s[i:end].strip()
        if len(piece) >= min_chunk_chars:
            chunks.append((i, end, piece))

        if end >= n:
            break
        # Advance with overlap to preserve context continuity.
        i = max(i + 1, end - overlap_chars)

    if not chunks:
        # Fallback for very short input that didn't satisfy chunk conditions.
        stripped = s.strip()
        if stripped:
            chunks = [(0, len(s), stripped)]
    return chunks


def _bm25_idf(*, doc_count: int, doc_freq: int) -> float:
    # Standard BM25 IDF with +1 smoothing for stability.
    return math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


@dataclass(frozen=True)
class RAGChunk:
    chunk_id: int
    start: int
    end: int
    text: str
    tf: dict[str, int]
    length: int
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    score: float
    start: int
    end: int
    text: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class RAGIndex:
    version: int
    created_at: str
    source: dict[str, Any]
    chunk_chars: int
    overlap_chars: int
    min_chunk_chars: int
    k1: float
    b: float
    avgdl: float
    idf: dict[str, float]
    chunks: list[RAGChunk]

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        # Clamp requested result count to a safe range.
        top_k = max(1, min(int(top_k), 80))
        # Tokenize query once.
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        qtf = Counter(q_terms)
        avgdl = self.avgdl or 1.0

        # Score each chunk with BM25 and keep positive scores.
        scored: list[tuple[float, RAGChunk]] = []
        for ch in self.chunks:
            dl = float(ch.length or 0)
            if dl <= 0:
                continue
            score = 0.0
            for term, qf in qtf.items():
                f = ch.tf.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                score += (idf * (f * (self.k1 + 1.0)) / (denom or 1.0)) * float(qf)
            if score > 0.0:
                scored.append((score, ch))

        # Highest score first.
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[RetrievedChunk] = []
        for score, ch in scored[:top_k]:
            out.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id,
                    score=float(score),
                    start=ch.start,
                    end=ch.end,
                    text=ch.text,
                    page_start=ch.page_start,
                    page_end=ch.page_end,
                )
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        # Serialize index into JSON-friendly primitives.
        return {
            "version": int(self.version),
            "created_at": self.created_at,
            "source": self.source,
            "chunking": {"chunk_chars": self.chunk_chars, "overlap_chars": self.overlap_chars, "min_chunk_chars": self.min_chunk_chars},
            "bm25": {"k1": self.k1, "b": self.b, "avgdl": self.avgdl},
            "idf": self.idf,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                    "tf": c.tf,
                    "length": c.length,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                }
                for c in self.chunks
            ],
        }

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "RAGIndex":
        # Validate index version first.
        try:
            version = int(obj.get("version") or 0)
        except Exception:
            version = 0
        if version != 1:
            raise RAGError(f"Unsupported index version: {version}")

        chunking = obj.get("chunking") or {}
        bm25 = obj.get("bm25") or {}

        # Parse chunk rows defensively (skip malformed entries).
        chunks_raw = obj.get("chunks") or []
        chunks: list[RAGChunk] = []
        for item in chunks_raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "")
            tf_raw = item.get("tf") or {}
            tf: dict[str, int] = {}
            if isinstance(tf_raw, dict):
                for k, v in tf_raw.items():
                    try:
                        tf[str(k)] = int(v)
                    except Exception:
                        continue
            # Optional page metadata for PDF-based citations.
            page_start: int | None = None
            page_end: int | None = None
            try:
                if item.get("page_start") is not None:
                    page_start = int(item.get("page_start"))
            except Exception:
                page_start = None
            try:
                if item.get("page_end") is not None:
                    page_end = int(item.get("page_end"))
            except Exception:
                page_end = None
            chunks.append(
                RAGChunk(
                    chunk_id=int(item.get("chunk_id") or 0),
                    start=int(item.get("start") or 0),
                    end=int(item.get("end") or 0),
                    text=text,
                    tf=tf,
                    length=int(item.get("length") or 0),
                    page_start=page_start,
                    page_end=page_end,
                )
            )

        # Parse IDF map defensively.
        idf_raw = obj.get("idf") or {}
        idf: dict[str, float] = {}
        if isinstance(idf_raw, dict):
            for k, v in idf_raw.items():
                try:
                    idf[str(k)] = float(v)
                except Exception:
                    continue

        return RAGIndex(
            version=1,
            created_at=str(obj.get("created_at") or ""),
            source=dict(obj.get("source") or {}),
            chunk_chars=int(chunking.get("chunk_chars") or 1200),
            overlap_chars=int(chunking.get("overlap_chars") or 200),
            min_chunk_chars=int(chunking.get("min_chunk_chars") or 200),
            k1=float(bm25.get("k1") or 1.5),
            b=float(bm25.get("b") or 0.75),
            avgdl=float(bm25.get("avgdl") or 1.0),
            idf=idf,
            chunks=chunks,
        )

    @staticmethod
    def load(path: str | Path) -> "RAGIndex":
        # Load and parse JSON index from disk.
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise RAGError(f"Index not found: {path}")
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise RAGError(f"Failed to read index: {path} ({e})") from e
        if not isinstance(obj, dict):
            raise RAGError(f"Invalid index file: {path}")
        return RAGIndex.from_dict(obj)

    def save(self, path: str | Path) -> None:
        # Persist index as UTF-8 JSON with stable formatting.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


# Build a BM25-backed index from raw document text.
def build_index_from_text(
    text: str,
    *,
    source: dict[str, Any] | None = None,
    page_spans: list[tuple[int, int, int]] | None = None,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
    min_chunk_chars: int = 200,
    k1: float = 1.5,
    b: float = 0.75,
) -> RAGIndex:
    # Build source metadata and include text length by default.
    src = dict(source or {})
    src.setdefault("text_chars", len(text or ""))

    # Output rows: (start, end, text, page_start, page_end).
    raw_chunks: list[tuple[int, int, str, int | None, int | None]] = []
    # Normalized page spans used when caller provides per-page mapping.
    normalized_spans: list[tuple[int, int, int]] = []
    if page_spans:
        for item in page_spans:
            try:
                start, end, page_num = int(item[0]), int(item[1]), int(item[2])
            except Exception:
                continue
            start = max(0, min(start, len(text or "")))
            end = max(0, min(end, len(text or "")))
            if end <= start or page_num <= 0:
                continue
            normalized_spans.append((start, end, page_num))
        # Keep page spans ordered to preserve original page sequence.
        normalized_spans.sort(key=lambda x: (x[0], x[1], x[2]))

    if normalized_spans:
        # Chunk each page span independently so chunk->page mapping stays exact.
        for span_start, span_end, page_num in normalized_spans:
            segment = (text or "")[span_start:span_end]
            for local_start, local_end, piece in _chunk_text(
                segment,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
                min_chunk_chars=min_chunk_chars,
            ):
                raw_chunks.append((span_start + local_start, span_start + local_end, piece, page_num, page_num))
    else:
        # No page metadata available: chunk the full text directly.
        for start, end, piece in _chunk_text(
            text,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            min_chunk_chars=min_chunk_chars,
        ):
            raw_chunks.append((start, end, piece, None, None))

    chunks: list[RAGChunk] = []
    df: Counter[str] = Counter()
    doc_lens: list[int] = []

    # Build chunk rows and aggregate DF/length stats for BM25.
    for start, end, piece, page_start, page_end in raw_chunks:
        tokens = _tokenize(piece)
        if not tokens:
            continue
        tf_counter = Counter(tokens)
        tf = dict(tf_counter)
        dl = int(sum(tf_counter.values()))
        doc_lens.append(dl)
        for term in tf_counter.keys():
            df[term] += 1
        chunks.append(
            RAGChunk(
                chunk_id=len(chunks) + 1,
                start=int(start),
                end=int(end),
                text=piece,
                tf=tf,
                length=dl,
                page_start=page_start,
                page_end=page_end,
            )
        )

    if not chunks:
        raise RAGError("No text chunks could be indexed.")

    # Compute corpus-level BM25 statistics.
    doc_count = len(chunks)
    avgdl = float(sum(doc_lens) / max(1, len(doc_lens)))
    idf = {term: _bm25_idf(doc_count=doc_count, doc_freq=freq) for term, freq in df.items()}

    # Stamp UTC creation time to help with index lifecycle/debugging.
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return RAGIndex(
        version=1,
        created_at=created_at,
        source=src,
        chunk_chars=int(chunk_chars),
        overlap_chars=int(overlap_chars),
        min_chunk_chars=int(min_chunk_chars),
        k1=float(k1),
        b=float(b),
        avgdl=avgdl,
        idf=idf,
        chunks=chunks,
    )
