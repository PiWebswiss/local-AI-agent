# file_readers.py — line-by-line explanation

Generated: 2026-02-18T09:56:40

Purpose: Local file text extraction (PDF/DOCX/PPTX/XLSX/HTML/text + image OCR via OCR.Space).

Format: each original source line is shown with its line number, followed by a short explanation.

```text
  1 from __future__ import annotations
      ↳ Imports annotations from the third-party module `__future__`.
  2 
      ↳ Blank line for readability.
  3 import codecs
      ↳ Imports standard library modules: codecs.
  4 import os
      ↳ Imports standard library modules: os.
  5 import re
      ↳ Imports standard library modules: re.
  6 from pathlib import Path
      ↳ Imports Path from the standard library module `pathlib`.
  7 
      ↳ Blank line for readability.
  8 import ocr_api
      ↳ Imports local project modules: ocr_api.
  9 
      ↳ Blank line for readability.
 10 
      ↳ Blank line for readability.
 11 class FileReadError(RuntimeError):
      ↳ Defines a custom exception class `FileReadError`.
 12     pass
      ↳ Control-flow keyword.
 13 
      ↳ Blank line for readability.
 14 
      ↳ Blank line for readability.
 15 def _truncate(text: str, *, max_chars: int) -> str:
      ↳ Defines function `_truncate()`.
 16     if max_chars <= 0:
      ↳ Conditional branch: checks a condition and chooses a code path.
 17         return text
      ↳ Returns a value from the current function.
 18     if len(text) <= max_chars:
      ↳ Conditional branch: checks a condition and chooses a code path.
 19         return text
      ↳ Returns a value from the current function.
 20     return text[: max_chars - 3].rstrip() + "..."
      ↳ Returns a value from the current function.
 21 
      ↳ Blank line for readability.
 22 
      ↳ Blank line for readability.
 23 def _is_probably_binary(data: bytes) -> bool:
      ↳ Defines function `_is_probably_binary()`.
 24     if not data:
      ↳ Conditional branch: checks a condition and chooses a code path.
 25         return False
      ↳ Returns a value from the current function.
 26     sample = data[:8000]
      ↳ Assignment: sets `sample`.
 27     if b"\x00" in sample:
      ↳ Conditional branch: checks a condition and chooses a code path.
 28         return True
      ↳ Returns a value from the current function.
 29     allowed = set(range(32, 127)) | {9, 10, 13}
      ↳ Assignment: sets `allowed`.
 30     printable = sum(1 for b in sample if b in allowed)
      ↳ Assignment: sets `printable`.
 31     ratio = printable / max(1, len(sample))
      ↳ Assignment: sets `ratio`.
 32     return ratio < 0.70
      ↳ Returns a value from the current function.
 33 
      ↳ Blank line for readability.
 34 
      ↳ Blank line for readability.
 35 def _decode_text_bytes(data: bytes) -> str:
      ↳ Defines function `_decode_text_bytes()`.
 36     if data.startswith(codecs.BOM_UTF8):
      ↳ Conditional branch: checks a condition and chooses a code path.
 37         return data.decode("utf-8-sig", errors="replace")
      ↳ Returns a value from the current function.
 38     if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
      ↳ Conditional branch: checks a condition and chooses a code path.
 39         return data.decode("utf-16", errors="replace")
      ↳ Returns a value from the current function.
 40     if data.startswith(codecs.BOM_UTF32_LE) or data.startswith(codecs.BOM_UTF32_BE):
      ↳ Conditional branch: checks a condition and chooses a code path.
 41         return data.decode("utf-32", errors="replace")
      ↳ Returns a value from the current function.
 42 
      ↳ Blank line for readability.
 43     for enc in ("utf-8", "utf-8-sig"):
      ↳ Loop: repeats the following block.
 44         try:
      ↳ Start of a `try` block for exception handling.
 45             return data.decode(enc)
      ↳ Returns a value from the current function.
 46         except UnicodeDecodeError:
      ↳ Exception handler: runs if the `try` block raises an error.
 47             pass
      ↳ Control-flow keyword.
 48     # Windows-friendly fallback (will decode any byte)
      ↳ Comment/documentation line.
 49     return data.decode("cp1252", errors="replace")
      ↳ Returns a value from the current function.
 50 
      ↳ Blank line for readability.
 51 
      ↳ Blank line for readability.
 52 def _read_bytes(path: Path, *, max_bytes: int) -> bytes:
      ↳ Defines function `_read_bytes()`.
 53     try:
      ↳ Start of a `try` block for exception handling.
 54         with path.open("rb") as f:
      ↳ Context manager block: ensures setup/teardown around a resource.
 55             if max_bytes and max_bytes > 0:
      ↳ Conditional branch: checks a condition and chooses a code path.
 56                 return f.read(max_bytes)
      ↳ Returns a value from the current function.
 57             return f.read()
      ↳ Returns a value from the current function.
 58     except OSError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
 59         raise FileReadError(f"Failed to read file: {path} ({e})") from e
      ↳ Raises an exception to signal an error.
 60 
      ↳ Blank line for readability.
 61 
      ↳ Blank line for readability.
 62 def _html_to_text(html: str) -> str:
      ↳ Defines function `_html_to_text()`.
 63     # Minimal extraction; good enough for local HTML files.
      ↳ Comment/documentation line.
 64     html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", html)
      ↳ Assignment: sets `html`.
 65     html = re.sub(r"(?i)<br\\s*/?>", "\n", html)
      ↳ Assignment: sets `html`.
 66     html = re.sub(r"(?i)</(p|div|li|tr|h\\d|blockquote)>", "\n", html)
      ↳ Assignment: sets `html`.
 67     html = re.sub(r"(?s)<[^>]+>", " ", html)
      ↳ Assignment: sets `html`.
 68     html = re.sub(r"[ \t]{2,}", " ", html)
      ↳ Assignment: sets `html`.
 69     html = re.sub(r"\n{3,}", "\n\n", html)
      ↳ Assignment: sets `html`.
 70     return html.strip()
      ↳ Returns a value from the current function.
 71 
      ↳ Blank line for readability.
 72 
      ↳ Blank line for readability.
 73 def _read_pdf(path: Path) -> str:
      ↳ Defines function `_read_pdf()`.
 74     try:
      ↳ Start of a `try` block for exception handling.
 75         from pypdf import PdfReader  # type: ignore
      ↳ Lazy/inner-scope imports PdfReader  # type: ignore from the third-party module `pypdf`.
 76     except Exception as e:  # pragma: no cover
      ↳ Exception handler: runs if the `try` block raises an error.
 77         raise FileReadError(f"PDF support requires 'pypdf'. ({e})") from e
      ↳ Raises an exception to signal an error.
 78 
      ↳ Blank line for readability.
 79     try:
      ↳ Start of a `try` block for exception handling.
 80         reader = PdfReader(str(path))
      ↳ Assignment: sets `reader`.
 81     except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
 82         raise FileReadError(f"Failed to open PDF: {path} ({e})") from e
      ↳ Raises an exception to signal an error.
 83 
      ↳ Blank line for readability.
 84     parts: list[str] = []
      ↳ Assignment: sets `parts: list[str]`.
 85     max_pages = int(os.getenv("AGENT_MAX_PDF_PAGES", "60"))
      ↳ Assignment: sets `max_pages`.
 86     for i, page in enumerate(reader.pages):
      ↳ Loop: repeats the following block.
 87         if max_pages > 0 and i >= max_pages:
      ↳ Conditional branch: checks a condition and chooses a code path.
 88             break
      ↳ Control-flow keyword.
 89         try:
      ↳ Start of a `try` block for exception handling.
 90             parts.append(page.extract_text() or "")
      ↳ Implementation detail: part of the surrounding logic.
 91         except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
 92             parts.append("")
      ↳ Implementation detail: part of the surrounding logic.
 93     return "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()
      ↳ Returns a value from the current function.
 94 
      ↳ Blank line for readability.
 95 
      ↳ Blank line for readability.
 96 def _read_docx(path: Path) -> str:
      ↳ Defines function `_read_docx()`.
 97     def _text_fallback() -> str | None:
      ↳ Defines function `_text_fallback()`.
 98         try:
      ↳ Start of a `try` block for exception handling.
 99             data = _read_bytes(path, max_bytes=int(os.getenv("AGENT_MAX_FILE_BYTES", "25000000")))
      ↳ Assignment: sets `data`.
100             if _is_probably_binary(data):
      ↳ Conditional branch: checks a condition and chooses a code path.
101                 return None
      ↳ Returns a value from the current function.
102             text = _decode_text_bytes(data).strip()
      ↳ Assignment: sets `text`.
103             return text or None
      ↳ Returns a value from the current function.
104         except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
105             return None
      ↳ Returns a value from the current function.
106 
      ↳ Blank line for readability.
107     try:
      ↳ Start of a `try` block for exception handling.
108         from docx import Document  # type: ignore
      ↳ Lazy/inner-scope imports Document  # type: ignore from the third-party module `docx`.
109     except Exception as e:  # pragma: no cover
      ↳ Exception handler: runs if the `try` block raises an error.
110         fallback = _text_fallback()
      ↳ Assignment: sets `fallback`.
111         if fallback is not None:
      ↳ Conditional branch: checks a condition and chooses a code path.
112             return fallback
      ↳ Returns a value from the current function.
113         raise FileReadError(f"DOCX support requires 'python-docx'. ({e})") from e
      ↳ Raises an exception to signal an error.
114 
      ↳ Blank line for readability.
115     try:
      ↳ Start of a `try` block for exception handling.
116         doc = Document(str(path))
      ↳ Assignment: sets `doc`.
117     except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
118         # Some users rename plain text files to .docx by mistake.
      ↳ Comment/documentation line.
119         # Fallback: if the content is text-like, decode and continue as plain text.
      ↳ Comment/documentation line.
120         fallback = _text_fallback()
      ↳ Assignment: sets `fallback`.
121         if fallback is not None:
      ↳ Conditional branch: checks a condition and chooses a code path.
122             return fallback
      ↳ Returns a value from the current function.
123         raise FileReadError(f"Failed to open DOCX: {path} ({e})") from e
      ↳ Raises an exception to signal an error.
124 
      ↳ Blank line for readability.
125     paras = [p.text for p in doc.paragraphs if (p.text or "").strip()]
      ↳ Assignment: sets `paras`.
126     return "\n".join(paras).strip()
      ↳ Returns a value from the current function.
127 
      ↳ Blank line for readability.
128 
      ↳ Blank line for readability.
129 def _read_pptx(path: Path) -> str:
      ↳ Defines function `_read_pptx()`.
130     try:
      ↳ Start of a `try` block for exception handling.
131         from pptx import Presentation  # type: ignore
      ↳ Lazy/inner-scope imports Presentation  # type: ignore from the third-party module `pptx`.
132     except Exception as e:  # pragma: no cover
      ↳ Exception handler: runs if the `try` block raises an error.
133         raise FileReadError(f"PPTX support requires 'python-pptx'. ({e})") from e
      ↳ Raises an exception to signal an error.
134 
      ↳ Blank line for readability.
135     try:
      ↳ Start of a `try` block for exception handling.
136         pres = Presentation(str(path))
      ↳ Assignment: sets `pres`.
137     except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
138         raise FileReadError(f"Failed to open PPTX: {path} ({e})") from e
      ↳ Raises an exception to signal an error.
139 
      ↳ Blank line for readability.
140     out: list[str] = []
      ↳ Assignment: sets `out: list[str]`.
141     max_slides = int(os.getenv("AGENT_MAX_PPTX_SLIDES", "80"))
      ↳ Assignment: sets `max_slides`.
142     for i, slide in enumerate(pres.slides):
      ↳ Loop: repeats the following block.
143         if max_slides > 0 and i >= max_slides:
      ↳ Conditional branch: checks a condition and chooses a code path.
144             break
      ↳ Control-flow keyword.
145         out.append(f"[Slide {i+1}]")
      ↳ Implementation detail: part of the surrounding logic.
146         for shape in slide.shapes:
      ↳ Loop: repeats the following block.
147             text = getattr(shape, "text", "") or ""
      ↳ Assignment: sets `text`.
148             text = text.strip()
      ↳ Assignment: sets `text`.
149             if text:
      ↳ Conditional branch: checks a condition and chooses a code path.
150                 out.append(text)
      ↳ Implementation detail: part of the surrounding logic.
151         out.append("")
      ↳ Implementation detail: part of the surrounding logic.
152     return "\n".join(out).strip()
      ↳ Returns a value from the current function.
153 
      ↳ Blank line for readability.
154 
      ↳ Blank line for readability.
155 def _read_xlsx(path: Path) -> str:
      ↳ Defines function `_read_xlsx()`.
156     try:
      ↳ Start of a `try` block for exception handling.
157         import openpyxl  # type: ignore
      ↳ Lazy/inner-scope imports third-party modules: openpyxl  # type: ignore.
158     except Exception as e:  # pragma: no cover
      ↳ Exception handler: runs if the `try` block raises an error.
159         raise FileReadError(f"XLSX support requires 'openpyxl'. ({e})") from e
      ↳ Raises an exception to signal an error.
160 
      ↳ Blank line for readability.
161     try:
      ↳ Start of a `try` block for exception handling.
162         wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
      ↳ Assignment: sets `wb`.
163     except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
164         raise FileReadError(f"Failed to open XLSX: {path} ({e})") from e
      ↳ Raises an exception to signal an error.
165 
      ↳ Blank line for readability.
166     max_sheets = int(os.getenv("AGENT_MAX_XLSX_SHEETS", "6"))
      ↳ Assignment: sets `max_sheets`.
167     max_rows = int(os.getenv("AGENT_MAX_XLSX_ROWS", "200"))
      ↳ Assignment: sets `max_rows`.
168     max_cols = int(os.getenv("AGENT_MAX_XLSX_COLS", "30"))
      ↳ Assignment: sets `max_cols`.
169 
      ↳ Blank line for readability.
170     out: list[str] = []
      ↳ Assignment: sets `out: list[str]`.
171     for si, name in enumerate(wb.sheetnames):
      ↳ Loop: repeats the following block.
172         if max_sheets > 0 and si >= max_sheets:
      ↳ Conditional branch: checks a condition and chooses a code path.
173             break
      ↳ Control-flow keyword.
174         ws = wb[name]
      ↳ Assignment: sets `ws`.
175         out.append(f"[Sheet] {name}")
      ↳ Implementation detail: part of the surrounding logic.
176         for ri, row in enumerate(ws.iter_rows(values_only=True)):
      ↳ Loop: repeats the following block.
177             if max_rows > 0 and ri >= max_rows:
      ↳ Conditional branch: checks a condition and chooses a code path.
178                 break
      ↳ Control-flow keyword.
179             cells = []
      ↳ Assignment: sets `cells`.
180             for ci, val in enumerate(row):
      ↳ Loop: repeats the following block.
181                 if max_cols > 0 and ci >= max_cols:
      ↳ Conditional branch: checks a condition and chooses a code path.
182                     break
      ↳ Control-flow keyword.
183                 if val is None:
      ↳ Conditional branch: checks a condition and chooses a code path.
184                     cells.append("")
      ↳ Implementation detail: part of the surrounding logic.
185                 else:
      ↳ Fallback branch for the preceding `if`/`elif`.
186                     cells.append(str(val))
      ↳ Implementation detail: part of the surrounding logic.
187             if any(c.strip() for c in cells):
      ↳ Conditional branch: checks a condition and chooses a code path.
188                 out.append("\t".join(cells).rstrip())
      ↳ Implementation detail: part of the surrounding logic.
189         out.append("")
      ↳ Implementation detail: part of the surrounding logic.
190     return "\n".join(out).strip()
      ↳ Returns a value from the current function.
191 
      ↳ Blank line for readability.
192 
      ↳ Blank line for readability.
193 def _read_image_ocr(path: Path) -> str:
      ↳ Defines function `_read_image_ocr()`.
194     data = _read_bytes(path, max_bytes=int(os.getenv("OCR_SPACE_MAX_BYTES", "8000000")))
      ↳ Assignment: sets `data`.
195     lang = os.getenv("OCR_SPACE_LANGUAGE") or os.getenv("OCR_SPACE_LANG")
      ↳ Assignment: sets `lang`.
196     try:
      ↳ Start of a `try` block for exception handling.
197         return ocr_api.ocr_image_bytes(data, filename=path.name, language=lang)
      ↳ Returns a value from the current function.
198     except ocr_api.OCRSpaceError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
199         raise FileReadError(str(e)) from e
      ↳ Raises an exception to signal an error.
200 
      ↳ Blank line for readability.
201 
      ↳ Blank line for readability.
202 def read_any_file(path: str, *, max_chars: int | None = None) -> str:
      ↳ Defines `read_any_file()`: Best-effort text extraction for many local file types.
203     """
      ↳ Implementation detail: part of the surrounding logic.
204     Best-effort text extraction from many file types.
      ↳ Implementation detail: part of the surrounding logic.
205 
      ↳ Blank line for readability.
206     Supported (when deps are installed): pdf, docx, pptx, xlsx, images (OCR).
      ↳ Implementation detail: part of the surrounding logic.
207     Falls back to decoding as text for unknown extensions.
      ↳ Implementation detail: part of the surrounding logic.
208     """
      ↳ Implementation detail: part of the surrounding logic.
209     p = Path(path)
      ↳ Assignment: sets `p`.
210     if not p.exists() or not p.is_file():
      ↳ Conditional branch: checks a condition and chooses a code path.
211         raise FileReadError(f"File not found: {path}")
      ↳ Raises an exception to signal an error.
212 
      ↳ Blank line for readability.
213     max_chars = int(os.getenv("AGENT_MAX_FILE_CHARS", "200000")) if max_chars is None else int(max_chars)
      ↳ Assignment: sets `max_chars`.
214     max_bytes = int(os.getenv("AGENT_MAX_FILE_BYTES", "25000000"))
      ↳ Assignment: sets `max_bytes`.
215 
      ↳ Blank line for readability.
216     ext = p.suffix.lower()
      ↳ Assignment: sets `ext`.
217     if ext == ".pdf":
      ↳ Conditional branch: checks a condition and chooses a code path.
218         return _truncate(_read_pdf(p), max_chars=max_chars)
      ↳ Returns a value from the current function.
219     if ext == ".docx":
      ↳ Conditional branch: checks a condition and chooses a code path.
220         return _truncate(_read_docx(p), max_chars=max_chars)
      ↳ Returns a value from the current function.
221     if ext == ".pptx":
      ↳ Conditional branch: checks a condition and chooses a code path.
222         return _truncate(_read_pptx(p), max_chars=max_chars)
      ↳ Returns a value from the current function.
223     if ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
      ↳ Conditional branch: checks a condition and chooses a code path.
224         return _truncate(_read_xlsx(p), max_chars=max_chars)
      ↳ Returns a value from the current function.
225     if ext in {".html", ".htm"}:
      ↳ Conditional branch: checks a condition and chooses a code path.
226         data = _read_bytes(p, max_bytes=max_bytes)
      ↳ Assignment: sets `data`.
227         if _is_probably_binary(data):
      ↳ Conditional branch: checks a condition and chooses a code path.
228             raise FileReadError("HTML file looks binary/unreadable.")
      ↳ Raises an exception to signal an error.
229         return _truncate(_html_to_text(_decode_text_bytes(data)), max_chars=max_chars)
      ↳ Returns a value from the current function.
230     if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
      ↳ Conditional branch: checks a condition and chooses a code path.
231         return _truncate(_read_image_ocr(p), max_chars=max_chars)
      ↳ Returns a value from the current function.
232 
      ↳ Blank line for readability.
233     data = _read_bytes(p, max_bytes=max_bytes)
      ↳ Assignment: sets `data`.
234     if _is_probably_binary(data):
      ↳ Conditional branch: checks a condition and chooses a code path.
235         raise FileReadError(f"Unsupported binary file type: {ext or '(no extension)'}")
      ↳ Raises an exception to signal an error.
236     return _truncate(_decode_text_bytes(data), max_chars=max_chars)
      ↳ Returns a value from the current function.
```
