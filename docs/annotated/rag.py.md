# rag.py — line-by-line explanation

Generated: 2026-02-18T09:56:40

Purpose: Simple local RAG: chunk text, build BM25 index, and retrieve relevant passages for Q&A.

Format: each original source line is shown with its line number, followed by a short explanation.

```text
  1 from __future__ import annotations
      ↳ Imports annotations from the third-party module `__future__`.
  2 
      ↳ Blank line for readability.
  3 import json
      ↳ Imports standard library modules: json.
  4 import math
      ↳ Imports standard library modules: math.
  5 import os
      ↳ Imports standard library modules: os.
  6 import re
      ↳ Imports standard library modules: re.
  7 from collections import Counter
      ↳ Imports Counter from the standard library module `collections`.
  8 from dataclasses import dataclass
      ↳ Imports dataclass from the standard library module `dataclasses`.
  9 from datetime import datetime, timezone
      ↳ Imports datetime, timezone from the standard library module `datetime`.
 10 from pathlib import Path
      ↳ Imports Path from the standard library module `pathlib`.
 11 from typing import Any
      ↳ Imports Any from the standard library module `typing`.
 12 
      ↳ Blank line for readability.
 13 
      ↳ Blank line for readability.
 14 class RAGError(RuntimeError):
      ↳ Defines a custom exception class `RAGError`.
 15     pass
      ↳ Control-flow keyword.
 16 
      ↳ Blank line for readability.
 17 
      ↳ Blank line for readability.
 18 _INDEX_EXT = ".rag.json"
      ↳ Assignment: sets `_INDEX_EXT`.
 19 _WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
      ↳ Assignment: sets `_WORD_RE`.
 20 
      ↳ Blank line for readability.
 21 
      ↳ Blank line for readability.
 22 def default_rag_dir() -> Path:
      ↳ Defines `default_rag_dir()`: Get the default directory for indexes (env `AGENT_RAG_DIR` or `./rag`).
 23     return Path(os.getenv("AGENT_RAG_DIR", "rag")).resolve()
      ↳ Returns a value from the current function.
 24 
      ↳ Blank line for readability.
 25 
      ↳ Blank line for readability.
 26 def sanitize_index_name(name: str) -> str:
      ↳ Defines `sanitize_index_name()`: Make a safe filename-friendly index name.
 27     name = (name or "").strip()
      ↳ Assignment: sets `name`.
 28     if not name:
      ↳ Conditional branch: checks a condition and chooses a code path.
 29         raise RAGError("Index name is empty.")
      ↳ Raises an exception to signal an error.
 30     cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
      ↳ Assignment: sets `cleaned`.
 31     if not cleaned:
      ↳ Conditional branch: checks a condition and chooses a code path.
 32         raise RAGError(f"Invalid index name: {name!r}")
      ↳ Raises an exception to signal an error.
 33     return cleaned
      ↳ Returns a value from the current function.
 34 
      ↳ Blank line for readability.
 35 
      ↳ Blank line for readability.
 36 def index_path(name: str, *, rag_dir: str | Path | None = None) -> Path:
      ↳ Defines `index_path()`: Compute the path to an index file for a given name.
 37     base = Path(rag_dir).resolve() if rag_dir is not None else default_rag_dir()
      ↳ Assignment: sets `base`.
 38     base.mkdir(parents=True, exist_ok=True)
      ↳ Assignment: sets `base.mkdir(parents`.
 39     safe = sanitize_index_name(name)
      ↳ Assignment: sets `safe`.
 40     return base / f"{safe}{_INDEX_EXT}"
      ↳ Returns a value from the current function.
 41 
      ↳ Blank line for readability.
 42 
      ↳ Blank line for readability.
 43 def list_indexes(*, rag_dir: str | Path | None = None) -> list[str]:
      ↳ Defines `list_indexes()`: List available book indexes in the index directory.
 44     base = Path(rag_dir).resolve() if rag_dir is not None else default_rag_dir()
      ↳ Assignment: sets `base`.
 45     if not base.exists():
      ↳ Conditional branch: checks a condition and chooses a code path.
 46         return []
      ↳ Returns a value from the current function.
 47     out: list[str] = []
      ↳ Assignment: sets `out: list[str]`.
 48     for p in sorted(base.glob(f"*{_INDEX_EXT}")):
      ↳ Loop: repeats the following block.
 49         if p.is_file():
      ↳ Conditional branch: checks a condition and chooses a code path.
 50             out.append(p.name[: -len(_INDEX_EXT)])
      ↳ Implementation detail: part of the surrounding logic.
 51     return out
      ↳ Returns a value from the current function.
 52 
      ↳ Blank line for readability.
 53 
      ↳ Blank line for readability.
 54 def _tokenize(text: str) -> list[str]:
      ↳ Defines function `_tokenize()`.
 55     toks = [t.lower() for t in _WORD_RE.findall(text or "")]
      ↳ Assignment: sets `toks`.
 56     return [t for t in toks if len(t) >= 2]
      ↳ Returns a value from the current function.
 57 
      ↳ Blank line for readability.
 58 
      ↳ Blank line for readability.
 59 def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int, min_chunk_chars: int) -> list[tuple[int, int, str]]:
      ↳ Defines function `_chunk_text()`.
 60     if chunk_chars <= 0:
      ↳ Conditional branch: checks a condition and chooses a code path.
 61         raise RAGError("chunk_chars must be > 0")
      ↳ Raises an exception to signal an error.
 62     if overlap_chars < 0:
      ↳ Conditional branch: checks a condition and chooses a code path.
 63         raise RAGError("overlap_chars must be >= 0")
      ↳ Raises an exception to signal an error.
 64     if min_chunk_chars < 0:
      ↳ Conditional branch: checks a condition and chooses a code path.
 65         raise RAGError("min_chunk_chars must be >= 0")
      ↳ Raises an exception to signal an error.
 66     overlap_chars = min(overlap_chars, max(0, chunk_chars - 1))
      ↳ Assignment: sets `overlap_chars`.
 67 
      ↳ Blank line for readability.
 68     s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
      ↳ Assignment: sets `s`.
 69     s = re.sub(r"[ \t]+\n", "\n", s)
      ↳ Assignment: sets `s`.
 70     s = re.sub(r"\n{4,}", "\n\n\n", s)
      ↳ Assignment: sets `s`.
 71 
      ↳ Blank line for readability.
 72     chunks: list[tuple[int, int, str]] = []
      ↳ Assignment: sets `chunks: list[tuple[int, int, str]]`.
 73     i = 0
      ↳ Assignment: sets `i`.
 74     n = len(s)
      ↳ Assignment: sets `n`.
 75     while i < n:
      ↳ Loop: repeats the following block.
 76         target_end = min(n, i + chunk_chars)
      ↳ Assignment: sets `target_end`.
 77         end = target_end
      ↳ Assignment: sets `end`.
 78 
      ↳ Blank line for readability.
 79         window_start = i + int(chunk_chars * 0.6)
      ↳ Assignment: sets `window_start`.
 80         window_start = min(window_start, target_end)
      ↳ Assignment: sets `window_start`.
 81         candidates = [
      ↳ Assignment: sets `candidates`.
 82             s.rfind("\n\n", window_start, target_end),
      ↳ Implementation detail: part of the surrounding logic.
 83             s.rfind("\n", window_start, target_end),
      ↳ Implementation detail: part of the surrounding logic.
 84             s.rfind(". ", window_start, target_end),
      ↳ Implementation detail: part of the surrounding logic.
 85         ]
      ↳ Implementation detail: part of the surrounding logic.
 86         best = max(candidates)
      ↳ Assignment: sets `best`.
 87         if best != -1 and best > i:
      ↳ Conditional branch: checks a condition and chooses a code path.
 88             end = best + (2 if s.startswith(". ", best) else 0)
      ↳ Assignment: sets `end`.
 89             end = max(end, i + 1)
      ↳ Assignment: sets `end`.
 90 
      ↳ Blank line for readability.
 91         piece = s[i:end].strip()
      ↳ Assignment: sets `piece`.
 92         if len(piece) >= min_chunk_chars:
      ↳ Conditional branch: checks a condition and chooses a code path.
 93             chunks.append((i, end, piece))
      ↳ Implementation detail: part of the surrounding logic.
 94 
      ↳ Blank line for readability.
 95         if end >= n:
      ↳ Conditional branch: checks a condition and chooses a code path.
 96             break
      ↳ Control-flow keyword.
 97         i = max(i + 1, end - overlap_chars)
      ↳ Assignment: sets `i`.
 98 
      ↳ Blank line for readability.
 99     if not chunks:
      ↳ Conditional branch: checks a condition and chooses a code path.
100         stripped = s.strip()
      ↳ Assignment: sets `stripped`.
101         if stripped:
      ↳ Conditional branch: checks a condition and chooses a code path.
102             chunks = [(0, len(s), stripped)]
      ↳ Assignment: sets `chunks`.
103     return chunks
      ↳ Returns a value from the current function.
104 
      ↳ Blank line for readability.
105 
      ↳ Blank line for readability.
106 def _bm25_idf(*, doc_count: int, doc_freq: int) -> float:
      ↳ Defines function `_bm25_idf()`.
107     return math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
      ↳ Returns a value from the current function.
108 
      ↳ Blank line for readability.
109 
      ↳ Blank line for readability.
110 @dataclass(frozen=True)
      ↳ Decorator line: modifies the behavior of the next function/method.
111 class RAGChunk:
      ↳ Defines a class `RAGChunk`.
112     chunk_id: int
      ↳ Implementation detail: part of the surrounding logic.
113     start: int
      ↳ Implementation detail: part of the surrounding logic.
114     end: int
      ↳ Implementation detail: part of the surrounding logic.
115     text: str
      ↳ Implementation detail: part of the surrounding logic.
116     tf: dict[str, int]
      ↳ Implementation detail: part of the surrounding logic.
117     length: int
      ↳ Implementation detail: part of the surrounding logic.
118 
      ↳ Blank line for readability.
119 
      ↳ Blank line for readability.
120 @dataclass(frozen=True)
      ↳ Decorator line: modifies the behavior of the next function/method.
121 class RetrievedChunk:
      ↳ Defines a class `RetrievedChunk`.
122     chunk_id: int
      ↳ Implementation detail: part of the surrounding logic.
123     score: float
      ↳ Implementation detail: part of the surrounding logic.
124     start: int
      ↳ Implementation detail: part of the surrounding logic.
125     end: int
      ↳ Implementation detail: part of the surrounding logic.
126     text: str
      ↳ Implementation detail: part of the surrounding logic.
127 
      ↳ Blank line for readability.
128 
      ↳ Blank line for readability.
129 @dataclass
      ↳ Decorator line: modifies the behavior of the next function/method.
130 class RAGIndex:
      ↳ Defines a class `RAGIndex`.
131     version: int
      ↳ Implementation detail: part of the surrounding logic.
132     created_at: str
      ↳ Implementation detail: part of the surrounding logic.
133     source: dict[str, Any]
      ↳ Implementation detail: part of the surrounding logic.
134     chunk_chars: int
      ↳ Implementation detail: part of the surrounding logic.
135     overlap_chars: int
      ↳ Implementation detail: part of the surrounding logic.
136     min_chunk_chars: int
      ↳ Implementation detail: part of the surrounding logic.
137     k1: float
      ↳ Implementation detail: part of the surrounding logic.
138     b: float
      ↳ Implementation detail: part of the surrounding logic.
139     avgdl: float
      ↳ Implementation detail: part of the surrounding logic.
140     idf: dict[str, float]
      ↳ Implementation detail: part of the surrounding logic.
141     chunks: list[RAGChunk]
      ↳ Implementation detail: part of the surrounding logic.
142 
      ↳ Blank line for readability.
143     def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
      ↳ Defines function `search()`.
144         top_k = max(1, min(int(top_k), 20))
      ↳ Assignment: sets `top_k`.
145         q_terms = _tokenize(query)
      ↳ Assignment: sets `q_terms`.
146         if not q_terms:
      ↳ Conditional branch: checks a condition and chooses a code path.
147             return []
      ↳ Returns a value from the current function.
148 
      ↳ Blank line for readability.
149         qtf = Counter(q_terms)
      ↳ Assignment: sets `qtf`.
150         avgdl = self.avgdl or 1.0
      ↳ Assignment: sets `avgdl`.
151 
      ↳ Blank line for readability.
152         scored: list[tuple[float, RAGChunk]] = []
      ↳ Assignment: sets `scored: list[tuple[float, RAGChunk]]`.
153         for ch in self.chunks:
      ↳ Loop: repeats the following block.
154             dl = float(ch.length or 0)
      ↳ Assignment: sets `dl`.
155             if dl <= 0:
      ↳ Conditional branch: checks a condition and chooses a code path.
156                 continue
      ↳ Control-flow keyword.
157             score = 0.0
      ↳ Assignment: sets `score`.
158             for term, qf in qtf.items():
      ↳ Loop: repeats the following block.
159                 f = ch.tf.get(term)
      ↳ Assignment: sets `f`.
160                 if not f:
      ↳ Conditional branch: checks a condition and chooses a code path.
161                     continue
      ↳ Control-flow keyword.
162                 idf = self.idf.get(term, 0.0)
      ↳ Assignment: sets `idf`.
163                 denom = f + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
      ↳ Assignment: sets `denom`.
164                 score += (idf * (f * (self.k1 + 1.0)) / (denom or 1.0)) * float(qf)
      ↳ Assignment: sets `score +`.
165             if score > 0.0:
      ↳ Conditional branch: checks a condition and chooses a code path.
166                 scored.append((score, ch))
      ↳ Implementation detail: part of the surrounding logic.
167 
      ↳ Blank line for readability.
168         scored.sort(key=lambda x: x[0], reverse=True)
      ↳ Assignment: sets `scored.sort(key`.
169         out: list[RetrievedChunk] = []
      ↳ Assignment: sets `out: list[RetrievedChunk]`.
170         for score, ch in scored[:top_k]:
      ↳ Loop: repeats the following block.
171             out.append(RetrievedChunk(chunk_id=ch.chunk_id, score=float(score), start=ch.start, end=ch.end, text=ch.text))
      ↳ Assignment: sets `out.append(RetrievedChunk(chunk_id`.
172         return out
      ↳ Returns a value from the current function.
173 
      ↳ Blank line for readability.
174     def to_dict(self) -> dict[str, Any]:
      ↳ Defines function `to_dict()`.
175         return {
      ↳ Returns a value from the current function.
176             "version": int(self.version),
      ↳ Implementation detail: part of the surrounding logic.
177             "created_at": self.created_at,
      ↳ Implementation detail: part of the surrounding logic.
178             "source": self.source,
      ↳ Implementation detail: part of the surrounding logic.
179             "chunking": {"chunk_chars": self.chunk_chars, "overlap_chars": self.overlap_chars, "min_chunk_chars": self.min_chunk_chars},
      ↳ Implementation detail: part of the surrounding logic.
180             "bm25": {"k1": self.k1, "b": self.b, "avgdl": self.avgdl},
      ↳ Implementation detail: part of the surrounding logic.
181             "idf": self.idf,
      ↳ Implementation detail: part of the surrounding logic.
182             "chunks": [
      ↳ Implementation detail: part of the surrounding logic.
183                 {"chunk_id": c.chunk_id, "start": c.start, "end": c.end, "text": c.text, "tf": c.tf, "length": c.length}
      ↳ Implementation detail: part of the surrounding logic.
184                 for c in self.chunks
      ↳ Loop: repeats the following block.
185             ],
      ↳ Implementation detail: part of the surrounding logic.
186         }
      ↳ Implementation detail: part of the surrounding logic.
187 
      ↳ Blank line for readability.
188     @staticmethod
      ↳ Decorator line: modifies the behavior of the next function/method.
189     def from_dict(obj: dict[str, Any]) -> "RAGIndex":
      ↳ Defines function `from_dict()`.
190         try:
      ↳ Start of a `try` block for exception handling.
191             version = int(obj.get("version") or 0)
      ↳ Assignment: sets `version`.
192         except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
193             version = 0
      ↳ Assignment: sets `version`.
194         if version != 1:
      ↳ Conditional branch: checks a condition and chooses a code path.
195             raise RAGError(f"Unsupported index version: {version}")
      ↳ Raises an exception to signal an error.
196 
      ↳ Blank line for readability.
197         chunking = obj.get("chunking") or {}
      ↳ Assignment: sets `chunking`.
198         bm25 = obj.get("bm25") or {}
      ↳ Assignment: sets `bm25`.
199 
      ↳ Blank line for readability.
200         chunks_raw = obj.get("chunks") or []
      ↳ Assignment: sets `chunks_raw`.
201         chunks: list[RAGChunk] = []
      ↳ Assignment: sets `chunks: list[RAGChunk]`.
202         for item in chunks_raw:
      ↳ Loop: repeats the following block.
203             if not isinstance(item, dict):
      ↳ Conditional branch: checks a condition and chooses a code path.
204                 continue
      ↳ Control-flow keyword.
205             text = str(item.get("text") or "")
      ↳ Assignment: sets `text`.
206             tf_raw = item.get("tf") or {}
      ↳ Assignment: sets `tf_raw`.
207             tf: dict[str, int] = {}
      ↳ Assignment: sets `tf: dict[str, int]`.
208             if isinstance(tf_raw, dict):
      ↳ Conditional branch: checks a condition and chooses a code path.
209                 for k, v in tf_raw.items():
      ↳ Loop: repeats the following block.
210                     try:
      ↳ Start of a `try` block for exception handling.
211                         tf[str(k)] = int(v)
      ↳ Assignment: sets `tf[str(k)]`.
212                     except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
213                         continue
      ↳ Control-flow keyword.
214             chunks.append(
      ↳ Implementation detail: part of the surrounding logic.
215                 RAGChunk(
      ↳ Implementation detail: part of the surrounding logic.
216                     chunk_id=int(item.get("chunk_id") or 0),
      ↳ Assignment: sets `chunk_id`.
217                     start=int(item.get("start") or 0),
      ↳ Assignment: sets `start`.
218                     end=int(item.get("end") or 0),
      ↳ Assignment: sets `end`.
219                     text=text,
      ↳ Assignment: sets `text`.
220                     tf=tf,
      ↳ Assignment: sets `tf`.
221                     length=int(item.get("length") or 0),
      ↳ Assignment: sets `length`.
222                 )
      ↳ Implementation detail: part of the surrounding logic.
223             )
      ↳ Implementation detail: part of the surrounding logic.
224 
      ↳ Blank line for readability.
225         idf_raw = obj.get("idf") or {}
      ↳ Assignment: sets `idf_raw`.
226         idf: dict[str, float] = {}
      ↳ Assignment: sets `idf: dict[str, float]`.
227         if isinstance(idf_raw, dict):
      ↳ Conditional branch: checks a condition and chooses a code path.
228             for k, v in idf_raw.items():
      ↳ Loop: repeats the following block.
229                 try:
      ↳ Start of a `try` block for exception handling.
230                     idf[str(k)] = float(v)
      ↳ Assignment: sets `idf[str(k)]`.
231                 except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
232                     continue
      ↳ Control-flow keyword.
233 
      ↳ Blank line for readability.
234         return RAGIndex(
      ↳ Returns a value from the current function.
235             version=1,
      ↳ Assignment: sets `version`.
236             created_at=str(obj.get("created_at") or ""),
      ↳ Assignment: sets `created_at`.
237             source=dict(obj.get("source") or {}),
      ↳ Assignment: sets `source`.
238             chunk_chars=int(chunking.get("chunk_chars") or 1200),
      ↳ Assignment: sets `chunk_chars`.
239             overlap_chars=int(chunking.get("overlap_chars") or 200),
      ↳ Assignment: sets `overlap_chars`.
240             min_chunk_chars=int(chunking.get("min_chunk_chars") or 200),
      ↳ Assignment: sets `min_chunk_chars`.
241             k1=float(bm25.get("k1") or 1.5),
      ↳ Assignment: sets `k1`.
242             b=float(bm25.get("b") or 0.75),
      ↳ Assignment: sets `b`.
243             avgdl=float(bm25.get("avgdl") or 1.0),
      ↳ Assignment: sets `avgdl`.
244             idf=idf,
      ↳ Assignment: sets `idf`.
245             chunks=chunks,
      ↳ Assignment: sets `chunks`.
246         )
      ↳ Implementation detail: part of the surrounding logic.
247 
      ↳ Blank line for readability.
248     @staticmethod
      ↳ Decorator line: modifies the behavior of the next function/method.
249     def load(path: str | Path) -> "RAGIndex":
      ↳ Defines function `load()`.
250         p = Path(path)
      ↳ Assignment: sets `p`.
251         if not p.exists() or not p.is_file():
      ↳ Conditional branch: checks a condition and chooses a code path.
252             raise RAGError(f"Index not found: {path}")
      ↳ Raises an exception to signal an error.
253         try:
      ↳ Start of a `try` block for exception handling.
254             obj = json.loads(p.read_text(encoding="utf-8"))
      ↳ Assignment: sets `obj`.
255         except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
256             raise RAGError(f"Failed to read index: {path} ({e})") from e
      ↳ Raises an exception to signal an error.
257         if not isinstance(obj, dict):
      ↳ Conditional branch: checks a condition and chooses a code path.
258             raise RAGError(f"Invalid index file: {path}")
      ↳ Raises an exception to signal an error.
259         return RAGIndex.from_dict(obj)
      ↳ Returns a value from the current function.
260 
      ↳ Blank line for readability.
261     def save(self, path: str | Path) -> None:
      ↳ Defines function `save()`.
262         p = Path(path)
      ↳ Assignment: sets `p`.
263         p.parent.mkdir(parents=True, exist_ok=True)
      ↳ Assignment: sets `p.parent.mkdir(parents`.
264         p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
      ↳ Assignment: sets `p.write_text(json.dumps(self.to_dict(), ensure_ascii`.
265 
      ↳ Blank line for readability.
266 
      ↳ Blank line for readability.
267 def build_index_from_text(
      ↳ Defines `build_index_from_text()`: Chunk a document and build a BM25 index for retrieval.
268     text: str,
      ↳ Implementation detail: part of the surrounding logic.
269     *,
      ↳ Implementation detail: part of the surrounding logic.
270     source: dict[str, Any] | None = None,
      ↳ Assignment: sets `source: dict[str, Any] | None`.
271     chunk_chars: int = 1200,
      ↳ Assignment: sets `chunk_chars: int`.
272     overlap_chars: int = 200,
      ↳ Assignment: sets `overlap_chars: int`.
273     min_chunk_chars: int = 200,
      ↳ Assignment: sets `min_chunk_chars: int`.
274     k1: float = 1.5,
      ↳ Assignment: sets `k1: float`.
275     b: float = 0.75,
      ↳ Assignment: sets `b: float`.
276 ) -> RAGIndex:
      ↳ Starts a new block (indented section) in Python.
277     src = dict(source or {})
      ↳ Assignment: sets `src`.
278     src.setdefault("text_chars", len(text or ""))
      ↳ Implementation detail: part of the surrounding logic.
279 
      ↳ Blank line for readability.
280     raw_chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars, min_chunk_chars=min_chunk_chars)
      ↳ Assignment: sets `raw_chunks`.
281     chunks: list[RAGChunk] = []
      ↳ Assignment: sets `chunks: list[RAGChunk]`.
282     df: Counter[str] = Counter()
      ↳ Assignment: sets `df: Counter[str]`.
283     doc_lens: list[int] = []
      ↳ Assignment: sets `doc_lens: list[int]`.
284 
      ↳ Blank line for readability.
285     for idx, (start, end, piece) in enumerate(raw_chunks):
      ↳ Loop: repeats the following block.
286         tokens = _tokenize(piece)
      ↳ Assignment: sets `tokens`.
287         if not tokens:
      ↳ Conditional branch: checks a condition and chooses a code path.
288             continue
      ↳ Control-flow keyword.
289         tf_counter = Counter(tokens)
      ↳ Assignment: sets `tf_counter`.
290         tf = dict(tf_counter)
      ↳ Assignment: sets `tf`.
291         dl = int(sum(tf_counter.values()))
      ↳ Assignment: sets `dl`.
292         doc_lens.append(dl)
      ↳ Implementation detail: part of the surrounding logic.
293         for term in tf_counter.keys():
      ↳ Loop: repeats the following block.
294             df[term] += 1
      ↳ Assignment: sets `df[term] +`.
295         chunks.append(RAGChunk(chunk_id=idx + 1, start=int(start), end=int(end), text=piece, tf=tf, length=dl))
      ↳ Assignment: sets `chunks.append(RAGChunk(chunk_id`.
296 
      ↳ Blank line for readability.
297     if not chunks:
      ↳ Conditional branch: checks a condition and chooses a code path.
298         raise RAGError("No text chunks could be indexed.")
      ↳ Raises an exception to signal an error.
299 
      ↳ Blank line for readability.
300     doc_count = len(chunks)
      ↳ Assignment: sets `doc_count`.
301     avgdl = float(sum(doc_lens) / max(1, len(doc_lens)))
      ↳ Assignment: sets `avgdl`.
302     idf = {term: _bm25_idf(doc_count=doc_count, doc_freq=freq) for term, freq in df.items()}
      ↳ Assignment: sets `idf`.
303 
      ↳ Blank line for readability.
304     created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
      ↳ Assignment: sets `created_at`.
305     return RAGIndex(
      ↳ Returns a value from the current function.
306         version=1,
      ↳ Assignment: sets `version`.
307         created_at=created_at,
      ↳ Assignment: sets `created_at`.
308         source=src,
      ↳ Assignment: sets `source`.
309         chunk_chars=int(chunk_chars),
      ↳ Assignment: sets `chunk_chars`.
310         overlap_chars=int(overlap_chars),
      ↳ Assignment: sets `overlap_chars`.
311         min_chunk_chars=int(min_chunk_chars),
      ↳ Assignment: sets `min_chunk_chars`.
312         k1=float(k1),
      ↳ Assignment: sets `k1`.
313         b=float(b),
      ↳ Assignment: sets `b`.
314         avgdl=avgdl,
      ↳ Assignment: sets `avgdl`.
315         idf=idf,
      ↳ Assignment: sets `idf`.
316         chunks=chunks,
      ↳ Assignment: sets `chunks`.
317     )
      ↳ Implementation detail: part of the surrounding logic.
318 
      ↳ Blank line for readability.
```
