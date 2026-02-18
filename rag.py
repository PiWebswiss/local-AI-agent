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
    pass


_INDEX_EXT = ".rag.json"
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def default_rag_dir() -> Path:
    return Path(os.getenv("AGENT_RAG_DIR", "rag")).resolve()


def sanitize_index_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise RAGError("Index name is empty.")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    if not cleaned:
        raise RAGError(f"Invalid index name: {name!r}")
    return cleaned


def index_path(name: str, *, rag_dir: str | Path | None = None) -> Path:
    base = Path(rag_dir).resolve() if rag_dir is not None else default_rag_dir()
    base.mkdir(parents=True, exist_ok=True)
    safe = sanitize_index_name(name)
    return base / f"{safe}{_INDEX_EXT}"


def list_indexes(*, rag_dir: str | Path | None = None) -> list[str]:
    base = Path(rag_dir).resolve() if rag_dir is not None else default_rag_dir()
    if not base.exists():
        return []
    out: list[str] = []
    for p in sorted(base.glob(f"*{_INDEX_EXT}")):
        if p.is_file():
            out.append(p.name[: -len(_INDEX_EXT)])
    return out


def _tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in _WORD_RE.findall(text or "")]
    return [t for t in toks if len(t) >= 2]


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int, min_chunk_chars: int) -> list[tuple[int, int, str]]:
    if chunk_chars <= 0:
        raise RAGError("chunk_chars must be > 0")
    if overlap_chars < 0:
        raise RAGError("overlap_chars must be >= 0")
    if min_chunk_chars < 0:
        raise RAGError("min_chunk_chars must be >= 0")
    overlap_chars = min(overlap_chars, max(0, chunk_chars - 1))

    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)

    chunks: list[tuple[int, int, str]] = []
    i = 0
    n = len(s)
    while i < n:
        target_end = min(n, i + chunk_chars)
        end = target_end

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

        piece = s[i:end].strip()
        if len(piece) >= min_chunk_chars:
            chunks.append((i, end, piece))

        if end >= n:
            break
        i = max(i + 1, end - overlap_chars)

    if not chunks:
        stripped = s.strip()
        if stripped:
            chunks = [(0, len(s), stripped)]
    return chunks


def _bm25_idf(*, doc_count: int, doc_freq: int) -> float:
    return math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


@dataclass(frozen=True)
class RAGChunk:
    chunk_id: int
    start: int
    end: int
    text: str
    tf: dict[str, int]
    length: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    score: float
    start: int
    end: int
    text: str


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
        top_k = max(1, min(int(top_k), 20))
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        qtf = Counter(q_terms)
        avgdl = self.avgdl or 1.0

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

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[RetrievedChunk] = []
        for score, ch in scored[:top_k]:
            out.append(RetrievedChunk(chunk_id=ch.chunk_id, score=float(score), start=ch.start, end=ch.end, text=ch.text))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "created_at": self.created_at,
            "source": self.source,
            "chunking": {"chunk_chars": self.chunk_chars, "overlap_chars": self.overlap_chars, "min_chunk_chars": self.min_chunk_chars},
            "bm25": {"k1": self.k1, "b": self.b, "avgdl": self.avgdl},
            "idf": self.idf,
            "chunks": [
                {"chunk_id": c.chunk_id, "start": c.start, "end": c.end, "text": c.text, "tf": c.tf, "length": c.length}
                for c in self.chunks
            ],
        }

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> "RAGIndex":
        try:
            version = int(obj.get("version") or 0)
        except Exception:
            version = 0
        if version != 1:
            raise RAGError(f"Unsupported index version: {version}")

        chunking = obj.get("chunking") or {}
        bm25 = obj.get("bm25") or {}

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
            chunks.append(
                RAGChunk(
                    chunk_id=int(item.get("chunk_id") or 0),
                    start=int(item.get("start") or 0),
                    end=int(item.get("end") or 0),
                    text=text,
                    tf=tf,
                    length=int(item.get("length") or 0),
                )
            )

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
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def build_index_from_text(
    text: str,
    *,
    source: dict[str, Any] | None = None,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
    min_chunk_chars: int = 200,
    k1: float = 1.5,
    b: float = 0.75,
) -> RAGIndex:
    src = dict(source or {})
    src.setdefault("text_chars", len(text or ""))

    raw_chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars, min_chunk_chars=min_chunk_chars)
    chunks: list[RAGChunk] = []
    df: Counter[str] = Counter()
    doc_lens: list[int] = []

    for idx, (start, end, piece) in enumerate(raw_chunks):
        tokens = _tokenize(piece)
        if not tokens:
            continue
        tf_counter = Counter(tokens)
        tf = dict(tf_counter)
        dl = int(sum(tf_counter.values()))
        doc_lens.append(dl)
        for term in tf_counter.keys():
            df[term] += 1
        chunks.append(RAGChunk(chunk_id=idx + 1, start=int(start), end=int(end), text=piece, tf=tf, length=dl))

    if not chunks:
        raise RAGError("No text chunks could be indexed.")

    doc_count = len(chunks)
    avgdl = float(sum(doc_lens) / max(1, len(doc_lens)))
    idf = {term: _bm25_idf(doc_count=doc_count, doc_freq=freq) for term, freq in df.items()}

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

