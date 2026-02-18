# web_tools.py — line-by-line explanation

Generated: 2026-02-18T09:56:40

Purpose: HTTPS-only web utilities: search (DuckDuckGo HTML/Lite/Instant Answer) and fetch+extract text (HTML/PDF/DOCX/PPTX/XLSX/images via OCR.Space).

Format: each original source line is shown with its line number, followed by a short explanation.

```text
  1 from __future__ import annotations
      ↳ Imports annotations from the third-party module `__future__`.
  2 
      ↳ Blank line for readability.
  3 import json
      ↳ Imports standard library modules: json.
  4 import io
      ↳ Imports standard library modules: io.
  5 import os
      ↳ Imports standard library modules: os.
  6 import re
      ↳ Imports standard library modules: re.
  7 import urllib.error
      ↳ Imports standard library modules: urllib.error.
  8 import urllib.parse
      ↳ Imports standard library modules: urllib.parse.
  9 import urllib.request
      ↳ Imports standard library modules: urllib.request.
 10 from html.parser import HTMLParser
      ↳ Imports HTMLParser from the standard library module `html.parser`.
 11 from typing import Any
      ↳ Imports Any from the standard library module `typing`.
 12 
      ↳ Blank line for readability.
 13 import ocr_api
      ↳ Imports local project modules: ocr_api.
 14 
      ↳ Blank line for readability.
 15 
      ↳ Blank line for readability.
 16 class WebToolError(RuntimeError):
      ↳ Defines a custom exception class `WebToolError`.
 17     pass
      ↳ Control-flow keyword.
 18 
      ↳ Blank line for readability.
 19 
      ↳ Blank line for readability.
 20 def ensure_https_url(url: str) -> str:
      ↳ Defines `ensure_https_url()`: Normalize/upgrade URLs so only `https://` is allowed.
 21     url = (url or "").strip()
      ↳ Assignment: sets `url`.
 22     if not url:
      ↳ Conditional branch: checks a condition and chooses a code path.
 23         raise WebToolError("URL is empty.")
      ↳ Raises an exception to signal an error.
 24 
      ↳ Blank line for readability.
 25     parsed = urllib.parse.urlparse(url)
      ↳ Assignment: sets `parsed`.
 26     if not parsed.scheme:
      ↳ Conditional branch: checks a condition and chooses a code path.
 27         url = "https://" + url.lstrip("/")
      ↳ Assignment: sets `url`.
 28         parsed = urllib.parse.urlparse(url)
      ↳ Assignment: sets `parsed`.
 29 
      ↳ Blank line for readability.
 30     if parsed.scheme == "https":
      ↳ Conditional branch: checks a condition and chooses a code path.
 31         return url
      ↳ Returns a value from the current function.
 32 
      ↳ Blank line for readability.
 33     if parsed.scheme == "http":
      ↳ Conditional branch: checks a condition and chooses a code path.
 34         return urllib.parse.urlunparse(parsed._replace(scheme="https"))
      ↳ Returns a value from the current function.
 35 
      ↳ Blank line for readability.
 36     raise WebToolError("Only https URLs are allowed.")
      ↳ Raises an exception to signal an error.
 37 
      ↳ Blank line for readability.
 38 
      ↳ Blank line for readability.
 39 class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
      ↳ Defines a class `_HTTPSOnlyRedirectHandler`.
 40     def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
      ↳ Defines function `redirect_request()`.
 41         absolute = urllib.parse.urljoin(req.full_url, newurl)
      ↳ Assignment: sets `absolute`.
 42         absolute = ensure_https_url(absolute)
      ↳ Assignment: sets `absolute`.
 43         return super().redirect_request(req, fp, code, msg, headers, absolute)
      ↳ Returns a value from the current function.
 44 
      ↳ Blank line for readability.
 45 
      ↳ Blank line for readability.
 46 _OPENER = urllib.request.build_opener(_HTTPSOnlyRedirectHandler())
      ↳ Assignment: sets `_OPENER`.
 47 
      ↳ Blank line for readability.
 48 
      ↳ Blank line for readability.
 49 def _http_get_bytes(url: str, *, timeout_s: float = 20.0, max_bytes: int = 2_000_000) -> tuple[str, bytes]:
      ↳ Defines function `_http_get_bytes()`.
 50     req = urllib.request.Request(
      ↳ Assignment: sets `req`.
 51         url,
      ↳ Implementation detail: part of the surrounding logic.
 52         headers={
      ↳ Assignment: sets `headers`.
 53             "User-Agent": "Mozilla/5.0 (compatible; LocalAI-Agent/1.0)",
      ↳ Implementation detail: part of the surrounding logic.
 54             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      ↳ Assignment: sets `"Accept": "text/html,application/xhtml+xml,application/xml;q`.
 55         },
      ↳ Implementation detail: part of the surrounding logic.
 56     )
      ↳ Implementation detail: part of the surrounding logic.
 57 
      ↳ Blank line for readability.
 58     try:
      ↳ Start of a `try` block for exception handling.
 59         with _OPENER.open(req, timeout=timeout_s) as resp:
      ↳ Context manager block: ensures setup/teardown around a resource.
 60             content_type = resp.headers.get("Content-Type", "") or ""
      ↳ Assignment: sets `content_type`.
 61             chunks: list[bytes] = []
      ↳ Assignment: sets `chunks: list[bytes]`.
 62             total = 0
      ↳ Assignment: sets `total`.
 63             while True:
      ↳ Loop: repeats the following block.
 64                 chunk = resp.read(min(64_000, max_bytes - total))
      ↳ Assignment: sets `chunk`.
 65                 if not chunk:
      ↳ Conditional branch: checks a condition and chooses a code path.
 66                     break
      ↳ Control-flow keyword.
 67                 chunks.append(chunk)
      ↳ Implementation detail: part of the surrounding logic.
 68                 total += len(chunk)
      ↳ Assignment: sets `total +`.
 69                 if total >= max_bytes:
      ↳ Conditional branch: checks a condition and chooses a code path.
 70                     break
      ↳ Control-flow keyword.
 71     except urllib.error.HTTPError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
 72         raise WebToolError(f"HTTP {e.code} while fetching {url}") from e
      ↳ Raises an exception to signal an error.
 73     except urllib.error.URLError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
 74         raise WebToolError(f"Network error while fetching {url}: {e.reason}") from e
      ↳ Raises an exception to signal an error.
 75 
      ↳ Blank line for readability.
 76     raw = b"".join(chunks)
      ↳ Assignment: sets `raw`.
 77     return content_type.lower(), raw
      ↳ Returns a value from the current function.
 78 
      ↳ Blank line for readability.
 79 
      ↳ Blank line for readability.
 80 def _http_get(url: str, *, timeout_s: float = 20.0, max_bytes: int = 2_000_000) -> tuple[str, str]:
      ↳ Defines function `_http_get()`.
 81     content_type, raw = _http_get_bytes(url, timeout_s=timeout_s, max_bytes=max_bytes)
      ↳ Assignment: sets `content_type, raw`.
 82     encoding = _guess_encoding(content_type, raw)
      ↳ Assignment: sets `encoding`.
 83     text = raw.decode(encoding, errors="replace")
      ↳ Assignment: sets `text`.
 84     return content_type, text
      ↳ Returns a value from the current function.
 85 
      ↳ Blank line for readability.
 86 
      ↳ Blank line for readability.
 87 def _guess_encoding(content_type: str, raw: bytes) -> str:
      ↳ Defines function `_guess_encoding()`.
 88     ct = content_type or ""
      ↳ Assignment: sets `ct`.
 89     match = re.search(r"charset=([^\s;]+)", ct, re.IGNORECASE)
      ↳ Assignment: sets `match`.
 90     if match:
      ↳ Conditional branch: checks a condition and chooses a code path.
 91         return match.group(1).strip("\"'").lower()
      ↳ Returns a value from the current function.
 92     # Try to sniff from HTML meta
      ↳ Comment/documentation line.
 93     head = raw[:50_000].decode("utf-8", errors="ignore")
      ↳ Assignment: sets `head`.
 94     match = re.search(r"charset\s*=\s*([a-zA-Z0-9_\-]+)", head, re.IGNORECASE)
      ↳ Assignment: sets `match`.
 95     if match:
      ↳ Conditional branch: checks a condition and chooses a code path.
 96         return match.group(1).lower()
      ↳ Returns a value from the current function.
 97     return "utf-8"
      ↳ Returns a value from the current function.
 98 
      ↳ Blank line for readability.
 99 
      ↳ Blank line for readability.
100 class _HTMLTextExtractor(HTMLParser):
      ↳ Defines a class `_HTMLTextExtractor`.
101     def __init__(self) -> None:
      ↳ Defines function `__init__()`.
102         super().__init__(convert_charrefs=True)
      ↳ Assignment: sets `super().__init__(convert_charrefs`.
103         self._out: list[str] = []
      ↳ Assignment: sets `self._out: list[str]`.
104         self._skip_depth = 0
      ↳ Assignment: sets `self._skip_depth`.
105 
      ↳ Blank line for readability.
106     def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
      ↳ Defines function `handle_starttag()`.
107         tag = tag.lower()
      ↳ Assignment: sets `tag`.
108         if tag in {"script", "style", "noscript"}:
      ↳ Conditional branch: checks a condition and chooses a code path.
109             self._skip_depth += 1
      ↳ Assignment: sets `self._skip_depth +`.
110             return
      ↳ Returns a value from the current function.
111         if self._skip_depth:
      ↳ Conditional branch: checks a condition and chooses a code path.
112             return
      ↳ Returns a value from the current function.
113         if tag in {"p", "div", "br", "hr", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}:
      ↳ Conditional branch: checks a condition and chooses a code path.
114             self._out.append("\n")
      ↳ Implementation detail: part of the surrounding logic.
115 
      ↳ Blank line for readability.
116     def handle_endtag(self, tag: str) -> None:
      ↳ Defines function `handle_endtag()`.
117         tag = tag.lower()
      ↳ Assignment: sets `tag`.
118         if tag in {"script", "style", "noscript"} and self._skip_depth:
      ↳ Conditional branch: checks a condition and chooses a code path.
119             self._skip_depth -= 1
      ↳ Assignment: sets `self._skip_depth -`.
120             return
      ↳ Returns a value from the current function.
121         if self._skip_depth:
      ↳ Conditional branch: checks a condition and chooses a code path.
122             return
      ↳ Returns a value from the current function.
123         if tag in {"p", "div", "li", "tr", "blockquote"}:
      ↳ Conditional branch: checks a condition and chooses a code path.
124             self._out.append("\n")
      ↳ Implementation detail: part of the surrounding logic.
125 
      ↳ Blank line for readability.
126     def handle_data(self, data: str) -> None:
      ↳ Defines function `handle_data()`.
127         if self._skip_depth:
      ↳ Conditional branch: checks a condition and chooses a code path.
128             return
      ↳ Returns a value from the current function.
129         s = (data or "").strip()
      ↳ Assignment: sets `s`.
130         if not s:
      ↳ Conditional branch: checks a condition and chooses a code path.
131             return
      ↳ Returns a value from the current function.
132         self._out.append(s)
      ↳ Implementation detail: part of the surrounding logic.
133         self._out.append(" ")
      ↳ Implementation detail: part of the surrounding logic.
134 
      ↳ Blank line for readability.
135     def text(self) -> str:
      ↳ Defines function `text()`.
136         raw = "".join(self._out)
      ↳ Assignment: sets `raw`.
137         raw = re.sub(r"[ \t]+\n", "\n", raw)
      ↳ Assignment: sets `raw`.
138         raw = re.sub(r"\n{3,}", "\n\n", raw)
      ↳ Assignment: sets `raw`.
139         raw = re.sub(r"[ \t]{2,}", " ", raw)
      ↳ Assignment: sets `raw`.
140         return raw.strip()
      ↳ Returns a value from the current function.
141 
      ↳ Blank line for readability.
142 
      ↳ Blank line for readability.
143 def fetch_url(url: str, *, timeout_s: float = 20.0, max_chars: int = 8_000) -> str:
      ↳ Defines `fetch_url()`: Fetch a URL and extract readable text based on content type/extension.
144     url = ensure_https_url(url)
      ↳ Assignment: sets `url`.
145     content_type, raw = _http_get_bytes(url, timeout_s=timeout_s, max_bytes=12_000_000)
      ↳ Assignment: sets `content_type, raw`.
146 
      ↳ Blank line for readability.
147     lowered = (content_type or "").lower()
      ↳ Assignment: sets `lowered`.
148     url_lower = url.lower()
      ↳ Assignment: sets `url_lower`.
149 
      ↳ Blank line for readability.
150     text = ""
      ↳ Assignment: sets `text`.
151     if "application/pdf" in lowered or url_lower.endswith(".pdf"):
      ↳ Conditional branch: checks a condition and chooses a code path.
152         try:
      ↳ Start of a `try` block for exception handling.
153             from pypdf import PdfReader  # type: ignore
      ↳ Lazy/inner-scope imports PdfReader  # type: ignore from the third-party module `pypdf`.
154 
      ↳ Blank line for readability.
155             reader = PdfReader(io.BytesIO(raw))
      ↳ Assignment: sets `reader`.
156             parts = []
      ↳ Assignment: sets `parts`.
157             for i, page in enumerate(reader.pages):
      ↳ Loop: repeats the following block.
158                 if i >= 60:
      ↳ Conditional branch: checks a condition and chooses a code path.
159                     break
      ↳ Control-flow keyword.
160                 try:
      ↳ Start of a `try` block for exception handling.
161                     parts.append(page.extract_text() or "")
      ↳ Implementation detail: part of the surrounding logic.
162                 except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
163                     parts.append("")
      ↳ Implementation detail: part of the surrounding logic.
164             text = "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()
      ↳ Assignment: sets `text`.
165         except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
166             raise WebToolError(f"Failed to extract PDF text from {url}: {e}") from e
      ↳ Raises an exception to signal an error.
167     elif (
      ↳ Conditional branch: checks a condition and chooses a code path.
168         "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in lowered
      ↳ Implementation detail: part of the surrounding logic.
169         or url_lower.endswith(".docx")
      ↳ Implementation detail: part of the surrounding logic.
170     ):
      ↳ Starts a new block (indented section) in Python.
171         try:
      ↳ Start of a `try` block for exception handling.
172             from docx import Document  # type: ignore
      ↳ Lazy/inner-scope imports Document  # type: ignore from the third-party module `docx`.
173 
      ↳ Blank line for readability.
174             doc = Document(io.BytesIO(raw))
      ↳ Assignment: sets `doc`.
175             paras = [p.text for p in doc.paragraphs if (p.text or "").strip()]
      ↳ Assignment: sets `paras`.
176             text = "\n".join(paras).strip()
      ↳ Assignment: sets `text`.
177         except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
178             raise WebToolError(f"Failed to extract DOCX text from {url}: {e}") from e
      ↳ Raises an exception to signal an error.
179     elif (
      ↳ Conditional branch: checks a condition and chooses a code path.
180         "application/vnd.openxmlformats-officedocument.presentationml.presentation" in lowered
      ↳ Implementation detail: part of the surrounding logic.
181         or url_lower.endswith(".pptx")
      ↳ Implementation detail: part of the surrounding logic.
182     ):
      ↳ Starts a new block (indented section) in Python.
183         try:
      ↳ Start of a `try` block for exception handling.
184             from pptx import Presentation  # type: ignore
      ↳ Lazy/inner-scope imports Presentation  # type: ignore from the third-party module `pptx`.
185 
      ↳ Blank line for readability.
186             pres = Presentation(io.BytesIO(raw))
      ↳ Assignment: sets `pres`.
187             out: list[str] = []
      ↳ Assignment: sets `out: list[str]`.
188             for si, slide in enumerate(pres.slides):
      ↳ Loop: repeats the following block.
189                 if si >= 80:
      ↳ Conditional branch: checks a condition and chooses a code path.
190                     break
      ↳ Control-flow keyword.
191                 out.append(f"[Slide {si+1}]")
      ↳ Implementation detail: part of the surrounding logic.
192                 for shape in slide.shapes:
      ↳ Loop: repeats the following block.
193                     s = (getattr(shape, "text", "") or "").strip()
      ↳ Assignment: sets `s`.
194                     if s:
      ↳ Conditional branch: checks a condition and chooses a code path.
195                         out.append(s)
      ↳ Implementation detail: part of the surrounding logic.
196                 out.append("")
      ↳ Implementation detail: part of the surrounding logic.
197             text = "\n".join(out).strip()
      ↳ Assignment: sets `text`.
198         except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
199             raise WebToolError(f"Failed to extract PPTX text from {url}: {e}") from e
      ↳ Raises an exception to signal an error.
200     elif (
      ↳ Conditional branch: checks a condition and chooses a code path.
201         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in lowered
      ↳ Implementation detail: part of the surrounding logic.
202         or url_lower.endswith(".xlsx")
      ↳ Implementation detail: part of the surrounding logic.
203         or url_lower.endswith(".xlsm")
      ↳ Implementation detail: part of the surrounding logic.
204     ):
      ↳ Starts a new block (indented section) in Python.
205         try:
      ↳ Start of a `try` block for exception handling.
206             import openpyxl  # type: ignore
      ↳ Lazy/inner-scope imports third-party modules: openpyxl  # type: ignore.
207 
      ↳ Blank line for readability.
208             wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True, read_only=True)
      ↳ Assignment: sets `wb`.
209             out: list[str] = []
      ↳ Assignment: sets `out: list[str]`.
210             for wi, name in enumerate(wb.sheetnames):
      ↳ Loop: repeats the following block.
211                 if wi >= 6:
      ↳ Conditional branch: checks a condition and chooses a code path.
212                     break
      ↳ Control-flow keyword.
213                 ws = wb[name]
      ↳ Assignment: sets `ws`.
214                 out.append(f"[Sheet] {name}")
      ↳ Implementation detail: part of the surrounding logic.
215                 for ri, row in enumerate(ws.iter_rows(values_only=True)):
      ↳ Loop: repeats the following block.
216                     if ri >= 200:
      ↳ Conditional branch: checks a condition and chooses a code path.
217                         break
      ↳ Control-flow keyword.
218                     cells = ["" if v is None else str(v) for v in row[:30]]
      ↳ Assignment: sets `cells`.
219                     if any(c.strip() for c in cells):
      ↳ Conditional branch: checks a condition and chooses a code path.
220                         out.append("\t".join(cells).rstrip())
      ↳ Implementation detail: part of the surrounding logic.
221                 out.append("")
      ↳ Implementation detail: part of the surrounding logic.
222             text = "\n".join(out).strip()
      ↳ Assignment: sets `text`.
223         except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
224             raise WebToolError(f"Failed to extract XLSX text from {url}: {e}") from e
      ↳ Raises an exception to signal an error.
225     elif lowered.startswith("image/") or any(
      ↳ Conditional branch: checks a condition and chooses a code path.
226         url_lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
      ↳ Implementation detail: part of the surrounding logic.
227     ):
      ↳ Starts a new block (indented section) in Python.
228         try:
      ↳ Start of a `try` block for exception handling.
229             lang = os.getenv("OCR_SPACE_LANGUAGE") or os.getenv("OCR_SPACE_LANG")
      ↳ Assignment: sets `lang`.
230             path = urllib.parse.urlparse(url).path or ""
      ↳ Assignment: sets `path`.
231             fname = (path.rsplit("/", 1)[-1] or "image").strip() or "image"
      ↳ Assignment: sets `fname`.
232             text = ocr_api.ocr_image_bytes(raw, filename=fname, language=lang)
      ↳ Assignment: sets `text`.
233         except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
234             raise WebToolError(f"Failed to OCR image from {url}: {e}") from e
      ↳ Raises an exception to signal an error.
235     else:
      ↳ Fallback branch for the preceding `if`/`elif`.
236         encoding = _guess_encoding(content_type, raw)
      ↳ Assignment: sets `encoding`.
237         text = raw.decode(encoding, errors="replace")
      ↳ Assignment: sets `text`.
238         if "text/html" in lowered or "<html" in text[:500].lower():
      ↳ Conditional branch: checks a condition and chooses a code path.
239             parser = _HTMLTextExtractor()
      ↳ Assignment: sets `parser`.
240             parser.feed(text)
      ↳ Implementation detail: part of the surrounding logic.
241             text = parser.text()
      ↳ Assignment: sets `text`.
242         else:
      ↳ Fallback branch for the preceding `if`/`elif`.
243             text = text.strip()
      ↳ Assignment: sets `text`.
244 
      ↳ Blank line for readability.
245     if len(text) > max_chars:
      ↳ Conditional branch: checks a condition and chooses a code path.
246         text = text[:max_chars].rstrip() + "..."
      ↳ Assignment: sets `text`.
247     return text
      ↳ Returns a value from the current function.
248 
      ↳ Blank line for readability.
249 
      ↳ Blank line for readability.
250 def _decode_ddg_href(href: str) -> str:
      ↳ Defines function `_decode_ddg_href()`.
251     href = (href or "").strip()
      ↳ Assignment: sets `href`.
252     if not href:
      ↳ Conditional branch: checks a condition and chooses a code path.
253         return ""
      ↳ Returns a value from the current function.
254     if href.startswith("//"):
      ↳ Conditional branch: checks a condition and chooses a code path.
255         href = "https:" + href
      ↳ Assignment: sets `href`.
256     if href.startswith("/"):
      ↳ Conditional branch: checks a condition and chooses a code path.
257         href = "https://duckduckgo.com" + href
      ↳ Assignment: sets `href`.
258     try:
      ↳ Start of a `try` block for exception handling.
259         parsed = urllib.parse.urlparse(href)
      ↳ Assignment: sets `parsed`.
260         path = (parsed.path or "").rstrip("/")
      ↳ Assignment: sets `path`.
261         last_seg = path.rsplit("/", 1)[-1] if path else ""
      ↳ Assignment: sets `last_seg`.
262         if parsed.netloc.endswith("duckduckgo.com") and last_seg == "l":
      ↳ Conditional branch: checks a condition and chooses a code path.
263             qs = urllib.parse.parse_qs(parsed.query)
      ↳ Assignment: sets `qs`.
264             if "uddg" in qs and qs["uddg"]:
      ↳ Conditional branch: checks a condition and chooses a code path.
265                 return urllib.parse.unquote(qs["uddg"][0])
      ↳ Returns a value from the current function.
266     except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
267         return href
      ↳ Returns a value from the current function.
268     return href
      ↳ Returns a value from the current function.
269 
      ↳ Blank line for readability.
270 
      ↳ Blank line for readability.
271 class _DuckDuckGoHTMLResults(HTMLParser):
      ↳ Defines a class `_DuckDuckGoHTMLResults`.
272     def __init__(self, *, max_results: int) -> None:
      ↳ Defines function `__init__()`.
273         super().__init__(convert_charrefs=True)
      ↳ Assignment: sets `super().__init__(convert_charrefs`.
274         self.max_results = max_results
      ↳ Assignment: sets `self.max_results`.
275         self.results: list[dict[str, str]] = []
      ↳ Assignment: sets `self.results: list[dict[str, str]]`.
276         self._capturing_title = False
      ↳ Assignment: sets `self._capturing_title`.
277         self._current: dict[str, str] | None = None
      ↳ Assignment: sets `self._current: dict[str, str] | None`.
278         self._title_parts: list[str] = []
      ↳ Assignment: sets `self._title_parts: list[str]`.
279 
      ↳ Blank line for readability.
280     def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
      ↳ Defines function `handle_starttag()`.
281         if len(self.results) >= self.max_results:
      ↳ Conditional branch: checks a condition and chooses a code path.
282             return
      ↳ Returns a value from the current function.
283         if tag.lower() != "a":
      ↳ Conditional branch: checks a condition and chooses a code path.
284             return
      ↳ Returns a value from the current function.
285         attr_map = {k.lower(): (v or "") for k, v in attrs}
      ↳ Assignment: sets `attr_map`.
286         cls = attr_map.get("class", "")
      ↳ Assignment: sets `cls`.
287         href = attr_map.get("href", "")
      ↳ Assignment: sets `href`.
288         if not href:
      ↳ Conditional branch: checks a condition and chooses a code path.
289             return
      ↳ Returns a value from the current function.
290         if "result__a" in cls or "result-link" in cls:
      ↳ Conditional branch: checks a condition and chooses a code path.
291             url = _decode_ddg_href(href)
      ↳ Assignment: sets `url`.
292             self._current = {"title": "", "url": url, "snippet": ""}
      ↳ Assignment: sets `self._current`.
293             self._title_parts = []
      ↳ Assignment: sets `self._title_parts`.
294             self._capturing_title = True
      ↳ Assignment: sets `self._capturing_title`.
295 
      ↳ Blank line for readability.
296     def handle_endtag(self, tag: str) -> None:
      ↳ Defines function `handle_endtag()`.
297         if tag.lower() != "a":
      ↳ Conditional branch: checks a condition and chooses a code path.
298             return
      ↳ Returns a value from the current function.
299         if self._capturing_title and self._current is not None:
      ↳ Conditional branch: checks a condition and chooses a code path.
300             title = " ".join([p.strip() for p in self._title_parts if p.strip()]).strip()
      ↳ Assignment: sets `title`.
301             title = re.sub(r"\s{2,}", " ", title)
      ↳ Assignment: sets `title`.
302             self._current["title"] = title
      ↳ Assignment: sets `self._current["title"]`.
303             if self._current.get("url") and self._current.get("title"):
      ↳ Conditional branch: checks a condition and chooses a code path.
304                 self.results.append(self._current)
      ↳ Implementation detail: part of the surrounding logic.
305             self._current = None
      ↳ Assignment: sets `self._current`.
306             self._title_parts = []
      ↳ Assignment: sets `self._title_parts`.
307             self._capturing_title = False
      ↳ Assignment: sets `self._capturing_title`.
308 
      ↳ Blank line for readability.
309     def handle_data(self, data: str) -> None:
      ↳ Defines function `handle_data()`.
310         if not self._capturing_title:
      ↳ Conditional branch: checks a condition and chooses a code path.
311             return
      ↳ Returns a value from the current function.
312         if self._current is None:
      ↳ Conditional branch: checks a condition and chooses a code path.
313             return
      ↳ Returns a value from the current function.
314         s = (data or "").strip()
      ↳ Assignment: sets `s`.
315         if s:
      ↳ Conditional branch: checks a condition and chooses a code path.
316             self._title_parts.append(s)
      ↳ Implementation detail: part of the surrounding logic.
317 
      ↳ Blank line for readability.
318 
      ↳ Blank line for readability.
319 class _DuckDuckGoLiteResults(HTMLParser):
      ↳ Defines a class `_DuckDuckGoLiteResults`.
320     def __init__(self, *, max_results: int) -> None:
      ↳ Defines function `__init__()`.
321         super().__init__(convert_charrefs=True)
      ↳ Assignment: sets `super().__init__(convert_charrefs`.
322         self.max_results = max_results
      ↳ Assignment: sets `self.max_results`.
323         self.results: list[dict[str, str]] = []
      ↳ Assignment: sets `self.results: list[dict[str, str]]`.
324         self._capturing_title = False
      ↳ Assignment: sets `self._capturing_title`.
325         self._current: dict[str, str] | None = None
      ↳ Assignment: sets `self._current: dict[str, str] | None`.
326         self._title_parts: list[str] = []
      ↳ Assignment: sets `self._title_parts: list[str]`.
327 
      ↳ Blank line for readability.
328     def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
      ↳ Defines function `handle_starttag()`.
329         if len(self.results) >= self.max_results:
      ↳ Conditional branch: checks a condition and chooses a code path.
330             return
      ↳ Returns a value from the current function.
331         if tag.lower() != "a":
      ↳ Conditional branch: checks a condition and chooses a code path.
332             return
      ↳ Returns a value from the current function.
333         attr_map = {k.lower(): (v or "") for k, v in attrs}
      ↳ Assignment: sets `attr_map`.
334         href = attr_map.get("href", "")
      ↳ Assignment: sets `href`.
335         if not href:
      ↳ Conditional branch: checks a condition and chooses a code path.
336             return
      ↳ Returns a value from the current function.
337 
      ↳ Blank line for readability.
338         url = _decode_ddg_href(href)
      ↳ Assignment: sets `url`.
339         try:
      ↳ Start of a `try` block for exception handling.
340             parsed = urllib.parse.urlparse(url)
      ↳ Assignment: sets `parsed`.
341         except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
342             return
      ↳ Returns a value from the current function.
343         if parsed.scheme not in {"http", "https"}:
      ↳ Conditional branch: checks a condition and chooses a code path.
344             return
      ↳ Returns a value from the current function.
345         if not parsed.netloc:
      ↳ Conditional branch: checks a condition and chooses a code path.
346             return
      ↳ Returns a value from the current function.
347         if parsed.netloc.endswith("duckduckgo.com"):
      ↳ Conditional branch: checks a condition and chooses a code path.
348             return
      ↳ Returns a value from the current function.
349 
      ↳ Blank line for readability.
350         self._current = {"title": "", "url": url, "snippet": ""}
      ↳ Assignment: sets `self._current`.
351         self._title_parts = []
      ↳ Assignment: sets `self._title_parts`.
352         self._capturing_title = True
      ↳ Assignment: sets `self._capturing_title`.
353 
      ↳ Blank line for readability.
354     def handle_endtag(self, tag: str) -> None:
      ↳ Defines function `handle_endtag()`.
355         if tag.lower() != "a":
      ↳ Conditional branch: checks a condition and chooses a code path.
356             return
      ↳ Returns a value from the current function.
357         if self._capturing_title and self._current is not None:
      ↳ Conditional branch: checks a condition and chooses a code path.
358             title = " ".join([p.strip() for p in self._title_parts if p.strip()]).strip()
      ↳ Assignment: sets `title`.
359             title = re.sub(r"\s{2,}", " ", title)
      ↳ Assignment: sets `title`.
360             self._current["title"] = title
      ↳ Assignment: sets `self._current["title"]`.
361             if self._current.get("url") and self._current.get("title"):
      ↳ Conditional branch: checks a condition and chooses a code path.
362                 self.results.append(self._current)
      ↳ Implementation detail: part of the surrounding logic.
363             self._current = None
      ↳ Assignment: sets `self._current`.
364             self._title_parts = []
      ↳ Assignment: sets `self._title_parts`.
365             self._capturing_title = False
      ↳ Assignment: sets `self._capturing_title`.
366 
      ↳ Blank line for readability.
367     def handle_data(self, data: str) -> None:
      ↳ Defines function `handle_data()`.
368         if not self._capturing_title:
      ↳ Conditional branch: checks a condition and chooses a code path.
369             return
      ↳ Returns a value from the current function.
370         if self._current is None:
      ↳ Conditional branch: checks a condition and chooses a code path.
371             return
      ↳ Returns a value from the current function.
372         s = (data or "").strip()
      ↳ Assignment: sets `s`.
373         if s:
      ↳ Conditional branch: checks a condition and chooses a code path.
374             self._title_parts.append(s)
      ↳ Implementation detail: part of the surrounding logic.
375 
      ↳ Blank line for readability.
376 
      ↳ Blank line for readability.
377 def _duckduckgo_html_search(query: str, *, max_results: int) -> list[dict[str, str]]:
      ↳ Defines function `_duckduckgo_html_search()`.
378     q = urllib.parse.quote_plus(query)
      ↳ Assignment: sets `q`.
379     url = f"https://duckduckgo.com/html/?q={q}"
      ↳ Assignment: sets `url`.
380     content_type, page = _http_get(url, timeout_s=20.0, max_bytes=1_500_000)
      ↳ Assignment: sets `content_type, page`.
381     if "text/html" not in content_type and "<html" not in page[:300].lower():
      ↳ Conditional branch: checks a condition and chooses a code path.
382         raise WebToolError("DuckDuckGo did not return HTML.")
      ↳ Raises an exception to signal an error.
383     parser = _DuckDuckGoHTMLResults(max_results=max_results)
      ↳ Assignment: sets `parser`.
384     parser.feed(page)
      ↳ Implementation detail: part of the surrounding logic.
385     return parser.results
      ↳ Returns a value from the current function.
386 
      ↳ Blank line for readability.
387 
      ↳ Blank line for readability.
388 def _duckduckgo_lite_search(query: str, *, max_results: int) -> list[dict[str, str]]:
      ↳ Defines function `_duckduckgo_lite_search()`.
389     q = urllib.parse.quote_plus(query)
      ↳ Assignment: sets `q`.
390     url = f"https://lite.duckduckgo.com/lite/?q={q}"
      ↳ Assignment: sets `url`.
391     content_type, page = _http_get(url, timeout_s=20.0, max_bytes=1_500_000)
      ↳ Assignment: sets `content_type, page`.
392     if "text/html" not in content_type and "<html" not in page[:300].lower():
      ↳ Conditional branch: checks a condition and chooses a code path.
393         raise WebToolError("DuckDuckGo Lite did not return HTML.")
      ↳ Raises an exception to signal an error.
394     parser = _DuckDuckGoLiteResults(max_results=max_results)
      ↳ Assignment: sets `parser`.
395     parser.feed(page)
      ↳ Implementation detail: part of the surrounding logic.
396     return parser.results
      ↳ Returns a value from the current function.
397 
      ↳ Blank line for readability.
398 
      ↳ Blank line for readability.
399 def _duckduckgo_instant_answer(query: str, *, max_results: int) -> list[dict[str, str]]:
      ↳ Defines function `_duckduckgo_instant_answer()`.
400     q = urllib.parse.quote_plus(query)
      ↳ Assignment: sets `q`.
401     url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
      ↳ Assignment: sets `url`.
402     _, text = _http_get(url, timeout_s=20.0, max_bytes=2_000_000)
      ↳ Assignment: sets `_, text`.
403     data = json.loads(text)
      ↳ Assignment: sets `data`.
404     out: list[dict[str, str]] = []
      ↳ Assignment: sets `out: list[dict[str, str]]`.
405 
      ↳ Blank line for readability.
406     abstract = (data.get("AbstractText") or "").strip()
      ↳ Assignment: sets `abstract`.
407     abstract_url = (data.get("AbstractURL") or "").strip()
      ↳ Assignment: sets `abstract_url`.
408     if abstract_url:
      ↳ Conditional branch: checks a condition and chooses a code path.
409         out.append({"title": (data.get("Heading") or "DuckDuckGo Abstract").strip(), "url": abstract_url, "snippet": abstract})
      ↳ Implementation detail: part of the surrounding logic.
410 
      ↳ Blank line for readability.
411     def add_topic(item: dict[str, Any]) -> None:
      ↳ Defines function `add_topic()`.
412         nonlocal out
      ↳ Implementation detail: part of the surrounding logic.
413         url2 = (item.get("FirstURL") or "").strip()
      ↳ Assignment: sets `url2`.
414         txt = (item.get("Text") or "").strip()
      ↳ Assignment: sets `txt`.
415         if url2 and txt:
      ↳ Conditional branch: checks a condition and chooses a code path.
416             out.append({"title": txt, "url": url2, "snippet": txt})
      ↳ Implementation detail: part of the surrounding logic.
417 
      ↳ Blank line for readability.
418     topics = data.get("RelatedTopics") or []
      ↳ Assignment: sets `topics`.
419     for item in topics:
      ↳ Loop: repeats the following block.
420         if len(out) >= max_results:
      ↳ Conditional branch: checks a condition and chooses a code path.
421             break
      ↳ Control-flow keyword.
422         if isinstance(item, dict) and "Topics" in item and isinstance(item["Topics"], list):
      ↳ Conditional branch: checks a condition and chooses a code path.
423             for sub in item["Topics"]:
      ↳ Loop: repeats the following block.
424                 if len(out) >= max_results:
      ↳ Conditional branch: checks a condition and chooses a code path.
425                     break
      ↳ Control-flow keyword.
426                 if isinstance(sub, dict):
      ↳ Conditional branch: checks a condition and chooses a code path.
427                     add_topic(sub)
      ↳ Implementation detail: part of the surrounding logic.
428         elif isinstance(item, dict):
      ↳ Conditional branch: checks a condition and chooses a code path.
429             add_topic(item)
      ↳ Implementation detail: part of the surrounding logic.
430 
      ↳ Blank line for readability.
431     return out[:max_results]
      ↳ Returns a value from the current function.
432 
      ↳ Blank line for readability.
433 
      ↳ Blank line for readability.
434 def web_search(query: str, *, max_results: int = 5) -> list[dict[str, str]]:
      ↳ Defines `web_search()`: DuckDuckGo search with multiple fallbacks; returns HTTPS-only result URLs.
435     query = (query or "").strip()
      ↳ Assignment: sets `query`.
436     if not query:
      ↳ Conditional branch: checks a condition and chooses a code path.
437         raise WebToolError("Query is empty.")
      ↳ Raises an exception to signal an error.
438     max_results = max(1, min(int(max_results), 10))
      ↳ Assignment: sets `max_results`.
439 
      ↳ Blank line for readability.
440     def normalize_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
      ↳ Defines function `normalize_results()`.
441         out: list[dict[str, str]] = []
      ↳ Assignment: sets `out: list[dict[str, str]]`.
442         seen: set[str] = set()
      ↳ Assignment: sets `seen: set[str]`.
443         for item in results:
      ↳ Loop: repeats the following block.
444             url = (item.get("url") or "").strip()
      ↳ Assignment: sets `url`.
445             if not url:
      ↳ Conditional branch: checks a condition and chooses a code path.
446                 continue
      ↳ Control-flow keyword.
447             url = _decode_ddg_href(url)
      ↳ Assignment: sets `url`.
448             try:
      ↳ Start of a `try` block for exception handling.
449                 url = ensure_https_url(url)
      ↳ Assignment: sets `url`.
450             except WebToolError:
      ↳ Exception handler: runs if the `try` block raises an error.
451                 continue
      ↳ Control-flow keyword.
452             try:
      ↳ Start of a `try` block for exception handling.
453                 parsed = urllib.parse.urlparse(url)
      ↳ Assignment: sets `parsed`.
454                 if parsed.netloc.endswith("duckduckgo.com"):
      ↳ Conditional branch: checks a condition and chooses a code path.
455                     continue
      ↳ Control-flow keyword.
456             except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
457                 continue
      ↳ Control-flow keyword.
458             if url in seen:
      ↳ Conditional branch: checks a condition and chooses a code path.
459                 continue
      ↳ Control-flow keyword.
460             seen.add(url)
      ↳ Implementation detail: part of the surrounding logic.
461             out.append(
      ↳ Implementation detail: part of the surrounding logic.
462                 {
      ↳ Implementation detail: part of the surrounding logic.
463                     "title": (item.get("title") or "").strip(),
      ↳ Implementation detail: part of the surrounding logic.
464                     "url": url,
      ↳ Implementation detail: part of the surrounding logic.
465                     "snippet": (item.get("snippet") or "").strip(),
      ↳ Implementation detail: part of the surrounding logic.
466                 }
      ↳ Implementation detail: part of the surrounding logic.
467             )
      ↳ Implementation detail: part of the surrounding logic.
468             if len(out) >= max_results:
      ↳ Conditional branch: checks a condition and chooses a code path.
469                 break
      ↳ Control-flow keyword.
470         return out
      ↳ Returns a value from the current function.
471 
      ↳ Blank line for readability.
472     results: list[dict[str, str]] = []
      ↳ Assignment: sets `results: list[dict[str, str]]`.
473     try:
      ↳ Start of a `try` block for exception handling.
474         results = normalize_results(_duckduckgo_html_search(query, max_results=max_results))
      ↳ Assignment: sets `results`.
475     except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
476         results = []
      ↳ Assignment: sets `results`.
477 
      ↳ Blank line for readability.
478     if results:
      ↳ Conditional branch: checks a condition and chooses a code path.
479         return results[:max_results]
      ↳ Returns a value from the current function.
480 
      ↳ Blank line for readability.
481     try:
      ↳ Start of a `try` block for exception handling.
482         results = normalize_results(_duckduckgo_lite_search(query, max_results=max_results))
      ↳ Assignment: sets `results`.
483     except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
484         results = []
      ↳ Assignment: sets `results`.
485 
      ↳ Blank line for readability.
486     if results:
      ↳ Conditional branch: checks a condition and chooses a code path.
487         return results[:max_results]
      ↳ Returns a value from the current function.
488 
      ↳ Blank line for readability.
489     try:
      ↳ Start of a `try` block for exception handling.
490         results = normalize_results(_duckduckgo_instant_answer(query, max_results=max_results))
      ↳ Assignment: sets `results`.
491         return results[:max_results]
      ↳ Returns a value from the current function.
492     except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
493         raise WebToolError(f"Search failed: {e}") from e
      ↳ Raises an exception to signal an error.
```
