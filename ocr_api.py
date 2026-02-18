# OCR.Space HTTPS client.
# Sends image bytes to OCR.Space and returns extracted text with strict HTTPS handling.
from __future__ import annotations

import json
import mimetypes
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request


class OCRSpaceError(RuntimeError):
    # Raised for OCR config, transport, and API response errors.
    pass


def _ensure_https_url(url: str) -> str:
    # Normalize whitespace and reject empty input.
    url = (url or "").strip()
    if not url:
        raise OCRSpaceError("URL is empty.")
    # Parse once to inspect scheme.
    parsed = urllib.parse.urlparse(url)
    # Default missing scheme to HTTPS.
    if not parsed.scheme:
        url = "https://" + url.lstrip("/")
        parsed = urllib.parse.urlparse(url)
    # Keep HTTPS unchanged.
    if parsed.scheme == "https":
        return url
    # Upgrade HTTP to HTTPS.
    if parsed.scheme == "http":
        return urllib.parse.urlunparse(parsed._replace(scheme="https"))
    # Reject non-web schemes.
    raise OCRSpaceError("Only https URLs are allowed.")


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        # Resolve relative redirect target.
        absolute = urllib.parse.urljoin(req.full_url, newurl)
        # Enforce HTTPS on redirect targets.
        absolute = _ensure_https_url(absolute)
        return super().redirect_request(req, fp, code, msg, headers, absolute)


# Shared opener used by OCR requests (inherits HTTPS-only redirect logic).
_OPENER = urllib.request.build_opener(_HTTPSOnlyRedirectHandler())


def _env(*names: str) -> str:
    # Return first non-empty environment variable from the provided names.
    for name in names:
        v = (os.getenv(name) or "").strip()
        if v:
            return v
    return ""


def _multipart_form(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    # Generate random multipart boundary to avoid collisions with payload content.
    boundary = "----LocalAIAgentBoundary" + secrets.token_hex(16)
    boundary_bytes = boundary.encode("ascii")

    body_parts: list[bytes] = []

    # Add scalar form fields.
    for name, value in fields.items():
        body_parts.append(b"--" + boundary_bytes + b"\r\n")
        body_parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body_parts.append((value or "").encode("utf-8"))
        body_parts.append(b"\r\n")

    # Add file parts.
    for name, (filename, content, content_type) in files.items():
        body_parts.append(b"--" + boundary_bytes + b"\r\n")
        body_parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body_parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body_parts.append(content)
        body_parts.append(b"\r\n")

    body_parts.append(b"--" + boundary_bytes + b"--\r\n")
    # Return encoded multipart body plus Content-Type header value.
    return b"".join(body_parts), f"multipart/form-data; boundary={boundary}"


# Main OCR helper used by file readers and web tools.
def ocr_image_bytes(
    data: bytes,
    *,
    filename: str = "image",
    language: str | None = None,
    timeout_s: float | None = None,
) -> str:
    """
    OCR via OCR.Space API (https://ocr.space/).

    Requires `OCR_SPACE_API_KEY` in the environment.
    Images are sent to a third-party service.
    """
    # API key is mandatory for OCR.Space requests.
    api_key = _env("OCR_SPACE_API_KEY")
    if not api_key:
        raise OCRSpaceError("OCR_SPACE_API_KEY is not set.")

    # Resolve endpoint and force HTTPS even if user passed HTTP.
    endpoint = _env("OCR_SPACE_URL") or "https://api.ocr.space/parse/image"
    endpoint = _ensure_https_url(endpoint)

    # Enforce upload-size safety cap.
    max_bytes = int(_env("OCR_SPACE_MAX_BYTES") or "8000000")
    if max_bytes > 0 and len(data) > max_bytes:
        raise OCRSpaceError(f"Image too large for OCR.Space ({len(data)} bytes > {max_bytes}).")

    # Pick OCR language and timeout from args/env.
    language = (language or _env("OCR_SPACE_LANGUAGE", "OCR_SPACE_LANG") or "eng").strip()
    timeout_s = float(timeout_s if timeout_s is not None else _env("OCR_SPACE_TIMEOUT_S") or "60")

    # Guess MIME type from filename; fallback to octet-stream.
    guessed = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    if "." not in filename:
        # best effort: if it's a common image signature, give it a better name
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            filename = filename + ".png"
            guessed = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            filename = filename + ".jpg"
            guessed = "image/jpeg"

    # Build OCR.Space form fields and file payload.
    fields = {
        "language": language,
        "isOverlayRequired": "false",
        "isTable": "false",
        "OCREngine": _env("OCR_SPACE_ENGINE") or "2",
    }
    files = {"file": (filename, data, guessed)}

    body, content_type = _multipart_form(fields, files)
    req = urllib.request.Request(
        endpoint,
        method="POST",
        data=body,
        headers={
            "apikey": api_key,
            "Content-Type": content_type,
            "Accept": "application/json",
        },
    )

    try:
        # Submit OCR request and read raw response.
        with _OPENER.open(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        # Preserve HTTP response body for clearer diagnostics.
        err = ""
        try:
            err = e.read().decode("utf-8", errors="replace")
        except Exception:
            err = str(e)
        raise OCRSpaceError(f"OCR.Space HTTP {e.code}: {err}") from e
    except urllib.error.URLError as e:
        raise OCRSpaceError(f"OCR.Space network error: {e.reason}") from e

    try:
        # Parse OCR.Space JSON payload.
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise OCRSpaceError(f"OCR.Space returned invalid JSON: {e}") from e

    # Convert API-level processing errors into one readable message.
    if obj.get("IsErroredOnProcessing"):
        msg = obj.get("ErrorMessage")
        if isinstance(msg, list):
            msg = "; ".join([str(x) for x in msg if str(x).strip()])
        msg = (str(msg) if msg is not None else "").strip()
        details = (str(obj.get("ErrorDetails") or "")).strip()
        raise OCRSpaceError(f"OCR.Space error: {msg or details or 'unknown error'}")

    # Extract parsed text blocks from response.
    parsed = obj.get("ParsedResults")
    if not isinstance(parsed, list) or not parsed:
        raise OCRSpaceError("OCR.Space response missing ParsedResults.")

    texts: list[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        t = (item.get("ParsedText") or "").strip()
        if t:
            texts.append(t)

    # Merge all text blocks in original order.
    return "\n\n".join(texts).strip()
