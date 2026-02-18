# ocr_api.py — line-by-line explanation

Generated: 2026-02-18T09:56:40

Purpose: OCR.Space HTTPS client for doing OCR on images.

Format: each original source line is shown with its line number, followed by a short explanation.

```text
  1 from __future__ import annotations
      ↳ Imports annotations from the third-party module `__future__`.
  2 
      ↳ Blank line for readability.
  3 import json
      ↳ Imports standard library modules: json.
  4 import mimetypes
      ↳ Imports standard library modules: mimetypes.
  5 import os
      ↳ Imports standard library modules: os.
  6 import secrets
      ↳ Imports standard library modules: secrets.
  7 import urllib.error
      ↳ Imports standard library modules: urllib.error.
  8 import urllib.parse
      ↳ Imports standard library modules: urllib.parse.
  9 import urllib.request
      ↳ Imports standard library modules: urllib.request.
 10 
      ↳ Blank line for readability.
 11 
      ↳ Blank line for readability.
 12 class OCRSpaceError(RuntimeError):
      ↳ Defines a custom exception class `OCRSpaceError`.
 13     pass
      ↳ Control-flow keyword.
 14 
      ↳ Blank line for readability.
 15 
      ↳ Blank line for readability.
 16 def _ensure_https_url(url: str) -> str:
      ↳ Defines function `_ensure_https_url()`.
 17     url = (url or "").strip()
      ↳ Assignment: sets `url`.
 18     if not url:
      ↳ Conditional branch: checks a condition and chooses a code path.
 19         raise OCRSpaceError("URL is empty.")
      ↳ Raises an exception to signal an error.
 20     parsed = urllib.parse.urlparse(url)
      ↳ Assignment: sets `parsed`.
 21     if not parsed.scheme:
      ↳ Conditional branch: checks a condition and chooses a code path.
 22         url = "https://" + url.lstrip("/")
      ↳ Assignment: sets `url`.
 23         parsed = urllib.parse.urlparse(url)
      ↳ Assignment: sets `parsed`.
 24     if parsed.scheme == "https":
      ↳ Conditional branch: checks a condition and chooses a code path.
 25         return url
      ↳ Returns a value from the current function.
 26     if parsed.scheme == "http":
      ↳ Conditional branch: checks a condition and chooses a code path.
 27         return urllib.parse.urlunparse(parsed._replace(scheme="https"))
      ↳ Returns a value from the current function.
 28     raise OCRSpaceError("Only https URLs are allowed.")
      ↳ Raises an exception to signal an error.
 29 
      ↳ Blank line for readability.
 30 
      ↳ Blank line for readability.
 31 class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
      ↳ Defines a class `_HTTPSOnlyRedirectHandler`.
 32     def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
      ↳ Defines function `redirect_request()`.
 33         absolute = urllib.parse.urljoin(req.full_url, newurl)
      ↳ Assignment: sets `absolute`.
 34         absolute = _ensure_https_url(absolute)
      ↳ Assignment: sets `absolute`.
 35         return super().redirect_request(req, fp, code, msg, headers, absolute)
      ↳ Returns a value from the current function.
 36 
      ↳ Blank line for readability.
 37 
      ↳ Blank line for readability.
 38 _OPENER = urllib.request.build_opener(_HTTPSOnlyRedirectHandler())
      ↳ Assignment: sets `_OPENER`.
 39 
      ↳ Blank line for readability.
 40 
      ↳ Blank line for readability.
 41 def _env(*names: str) -> str:
      ↳ Defines function `_env()`.
 42     for name in names:
      ↳ Loop: repeats the following block.
 43         v = (os.getenv(name) or "").strip()
      ↳ Assignment: sets `v`.
 44         if v:
      ↳ Conditional branch: checks a condition and chooses a code path.
 45             return v
      ↳ Returns a value from the current function.
 46     return ""
      ↳ Returns a value from the current function.
 47 
      ↳ Blank line for readability.
 48 
      ↳ Blank line for readability.
 49 def _multipart_form(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
      ↳ Defines function `_multipart_form()`.
 50     boundary = "----LocalAIAgentBoundary" + secrets.token_hex(16)
      ↳ Assignment: sets `boundary`.
 51     boundary_bytes = boundary.encode("ascii")
      ↳ Assignment: sets `boundary_bytes`.
 52 
      ↳ Blank line for readability.
 53     body_parts: list[bytes] = []
      ↳ Assignment: sets `body_parts: list[bytes]`.
 54 
      ↳ Blank line for readability.
 55     for name, value in fields.items():
      ↳ Loop: repeats the following block.
 56         body_parts.append(b"--" + boundary_bytes + b"\r\n")
      ↳ Implementation detail: part of the surrounding logic.
 57         body_parts.append(
      ↳ Implementation detail: part of the surrounding logic.
 58             f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
      ↳ Assignment: sets `f'Content-Disposition: form-data; name`.
 59         )
      ↳ Implementation detail: part of the surrounding logic.
 60         body_parts.append((value or "").encode("utf-8"))
      ↳ Implementation detail: part of the surrounding logic.
 61         body_parts.append(b"\r\n")
      ↳ Implementation detail: part of the surrounding logic.
 62 
      ↳ Blank line for readability.
 63     for name, (filename, content, content_type) in files.items():
      ↳ Loop: repeats the following block.
 64         body_parts.append(b"--" + boundary_bytes + b"\r\n")
      ↳ Implementation detail: part of the surrounding logic.
 65         body_parts.append(
      ↳ Implementation detail: part of the surrounding logic.
 66             f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
      ↳ Assignment: sets `f'Content-Disposition: form-data; name`.
 67         )
      ↳ Implementation detail: part of the surrounding logic.
 68         body_parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
      ↳ Implementation detail: part of the surrounding logic.
 69         body_parts.append(content)
      ↳ Implementation detail: part of the surrounding logic.
 70         body_parts.append(b"\r\n")
      ↳ Implementation detail: part of the surrounding logic.
 71 
      ↳ Blank line for readability.
 72     body_parts.append(b"--" + boundary_bytes + b"--\r\n")
      ↳ Implementation detail: part of the surrounding logic.
 73     return b"".join(body_parts), f"multipart/form-data; boundary={boundary}"
      ↳ Returns a value from the current function.
 74 
      ↳ Blank line for readability.
 75 
      ↳ Blank line for readability.
 76 def ocr_image_bytes(
      ↳ Defines `ocr_image_bytes()`: Send image bytes to OCR.Space and return extracted text.
 77     data: bytes,
      ↳ Implementation detail: part of the surrounding logic.
 78     *,
      ↳ Implementation detail: part of the surrounding logic.
 79     filename: str = "image",
      ↳ Assignment: sets `filename: str`.
 80     language: str | None = None,
      ↳ Assignment: sets `language: str | None`.
 81     timeout_s: float | None = None,
      ↳ Assignment: sets `timeout_s: float | None`.
 82 ) -> str:
      ↳ Starts a new block (indented section) in Python.
 83     """
      ↳ Implementation detail: part of the surrounding logic.
 84     OCR via OCR.Space API (https://ocr.space/).
      ↳ Implementation detail: part of the surrounding logic.
 85 
      ↳ Blank line for readability.
 86     Requires `OCR_SPACE_API_KEY` in the environment.
      ↳ Implementation detail: part of the surrounding logic.
 87     Images are sent to a third-party service.
      ↳ Implementation detail: part of the surrounding logic.
 88     """
      ↳ Implementation detail: part of the surrounding logic.
 89     api_key = _env("OCR_SPACE_API_KEY")
      ↳ Assignment: sets `api_key`.
 90     if not api_key:
      ↳ Conditional branch: checks a condition and chooses a code path.
 91         raise OCRSpaceError("OCR_SPACE_API_KEY is not set.")
      ↳ Raises an exception to signal an error.
 92 
      ↳ Blank line for readability.
 93     endpoint = _env("OCR_SPACE_URL") or "https://api.ocr.space/parse/image"
      ↳ Assignment: sets `endpoint`.
 94     endpoint = _ensure_https_url(endpoint)
      ↳ Assignment: sets `endpoint`.
 95 
      ↳ Blank line for readability.
 96     max_bytes = int(_env("OCR_SPACE_MAX_BYTES") or "8000000")
      ↳ Assignment: sets `max_bytes`.
 97     if max_bytes > 0 and len(data) > max_bytes:
      ↳ Conditional branch: checks a condition and chooses a code path.
 98         raise OCRSpaceError(f"Image too large for OCR.Space ({len(data)} bytes > {max_bytes}).")
      ↳ Raises an exception to signal an error.
 99 
      ↳ Blank line for readability.
100     language = (language or _env("OCR_SPACE_LANGUAGE", "OCR_SPACE_LANG") or "eng").strip()
      ↳ Assignment: sets `language`.
101     timeout_s = float(timeout_s if timeout_s is not None else _env("OCR_SPACE_TIMEOUT_S") or "60")
      ↳ Assignment: sets `timeout_s`.
102 
      ↳ Blank line for readability.
103     guessed = mimetypes.guess_type(filename)[0] or "application/octet-stream"
      ↳ Assignment: sets `guessed`.
104     if "." not in filename:
      ↳ Conditional branch: checks a condition and chooses a code path.
105         # best effort: if it's a common image signature, give it a better name
      ↳ Comment/documentation line.
106         if data.startswith(b"\x89PNG\r\n\x1a\n"):
      ↳ Conditional branch: checks a condition and chooses a code path.
107             filename = filename + ".png"
      ↳ Assignment: sets `filename`.
108             guessed = "image/png"
      ↳ Assignment: sets `guessed`.
109         elif data.startswith(b"\xff\xd8\xff"):
      ↳ Conditional branch: checks a condition and chooses a code path.
110             filename = filename + ".jpg"
      ↳ Assignment: sets `filename`.
111             guessed = "image/jpeg"
      ↳ Assignment: sets `guessed`.
112 
      ↳ Blank line for readability.
113     fields = {
      ↳ Assignment: sets `fields`.
114         "language": language,
      ↳ Implementation detail: part of the surrounding logic.
115         "isOverlayRequired": "false",
      ↳ Implementation detail: part of the surrounding logic.
116         "isTable": "false",
      ↳ Implementation detail: part of the surrounding logic.
117         "OCREngine": _env("OCR_SPACE_ENGINE") or "2",
      ↳ Implementation detail: part of the surrounding logic.
118     }
      ↳ Implementation detail: part of the surrounding logic.
119     files = {"file": (filename, data, guessed)}
      ↳ Assignment: sets `files`.
120 
      ↳ Blank line for readability.
121     body, content_type = _multipart_form(fields, files)
      ↳ Assignment: sets `body, content_type`.
122     req = urllib.request.Request(
      ↳ Assignment: sets `req`.
123         endpoint,
      ↳ Implementation detail: part of the surrounding logic.
124         method="POST",
      ↳ Assignment: sets `method`.
125         data=body,
      ↳ Assignment: sets `data`.
126         headers={
      ↳ Assignment: sets `headers`.
127             "apikey": api_key,
      ↳ Implementation detail: part of the surrounding logic.
128             "Content-Type": content_type,
      ↳ Implementation detail: part of the surrounding logic.
129             "Accept": "application/json",
      ↳ Implementation detail: part of the surrounding logic.
130         },
      ↳ Implementation detail: part of the surrounding logic.
131     )
      ↳ Implementation detail: part of the surrounding logic.
132 
      ↳ Blank line for readability.
133     try:
      ↳ Start of a `try` block for exception handling.
134         with _OPENER.open(req, timeout=timeout_s) as resp:
      ↳ Context manager block: ensures setup/teardown around a resource.
135             raw = resp.read()
      ↳ Assignment: sets `raw`.
136     except urllib.error.HTTPError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
137         err = ""
      ↳ Assignment: sets `err`.
138         try:
      ↳ Start of a `try` block for exception handling.
139             err = e.read().decode("utf-8", errors="replace")
      ↳ Assignment: sets `err`.
140         except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
141             err = str(e)
      ↳ Assignment: sets `err`.
142         raise OCRSpaceError(f"OCR.Space HTTP {e.code}: {err}") from e
      ↳ Raises an exception to signal an error.
143     except urllib.error.URLError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
144         raise OCRSpaceError(f"OCR.Space network error: {e.reason}") from e
      ↳ Raises an exception to signal an error.
145 
      ↳ Blank line for readability.
146     try:
      ↳ Start of a `try` block for exception handling.
147         obj = json.loads(raw.decode("utf-8", errors="replace"))
      ↳ Assignment: sets `obj`.
148     except json.JSONDecodeError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
149         raise OCRSpaceError(f"OCR.Space returned invalid JSON: {e}") from e
      ↳ Raises an exception to signal an error.
150 
      ↳ Blank line for readability.
151     if obj.get("IsErroredOnProcessing"):
      ↳ Conditional branch: checks a condition and chooses a code path.
152         msg = obj.get("ErrorMessage")
      ↳ Assignment: sets `msg`.
153         if isinstance(msg, list):
      ↳ Conditional branch: checks a condition and chooses a code path.
154             msg = "; ".join([str(x) for x in msg if str(x).strip()])
      ↳ Assignment: sets `msg`.
155         msg = (str(msg) if msg is not None else "").strip()
      ↳ Assignment: sets `msg`.
156         details = (str(obj.get("ErrorDetails") or "")).strip()
      ↳ Assignment: sets `details`.
157         raise OCRSpaceError(f"OCR.Space error: {msg or details or 'unknown error'}")
      ↳ Raises an exception to signal an error.
158 
      ↳ Blank line for readability.
159     parsed = obj.get("ParsedResults")
      ↳ Assignment: sets `parsed`.
160     if not isinstance(parsed, list) or not parsed:
      ↳ Conditional branch: checks a condition and chooses a code path.
161         raise OCRSpaceError("OCR.Space response missing ParsedResults.")
      ↳ Raises an exception to signal an error.
162 
      ↳ Blank line for readability.
163     texts: list[str] = []
      ↳ Assignment: sets `texts: list[str]`.
164     for item in parsed:
      ↳ Loop: repeats the following block.
165         if not isinstance(item, dict):
      ↳ Conditional branch: checks a condition and chooses a code path.
166             continue
      ↳ Control-flow keyword.
167         t = (item.get("ParsedText") or "").strip()
      ↳ Assignment: sets `t`.
168         if t:
      ↳ Conditional branch: checks a condition and chooses a code path.
169             texts.append(t)
      ↳ Implementation detail: part of the surrounding logic.
170 
      ↳ Blank line for readability.
171     return "\n\n".join(texts).strip()
      ↳ Returns a value from the current function.
```
