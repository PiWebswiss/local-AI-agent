from __future__ import annotations

import codecs
import os
import re
from pathlib import Path

import ocr_api


class FileReadError(RuntimeError):
    pass


def _truncate(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _is_probably_binary(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:8000]
    if b"\x00" in sample:
        return True
    allowed = set(range(32, 127)) | {9, 10, 13}
    printable = sum(1 for b in sample if b in allowed)
    ratio = printable / max(1, len(sample))
    return ratio < 0.70


def _decode_text_bytes(data: bytes) -> str:
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig", errors="replace")
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return data.decode("utf-16", errors="replace")
    if data.startswith(codecs.BOM_UTF32_LE) or data.startswith(codecs.BOM_UTF32_BE):
        return data.decode("utf-32", errors="replace")

    for enc in ("utf-8", "utf-8-sig"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    # Windows-friendly fallback (will decode any byte)
    return data.decode("cp1252", errors="replace")


def _read_bytes(path: Path, *, max_bytes: int) -> bytes:
    try:
        with path.open("rb") as f:
            if max_bytes and max_bytes > 0:
                return f.read(max_bytes)
            return f.read()
    except OSError as e:
        raise FileReadError(f"Failed to read file: {path} ({e})") from e


def _html_to_text(html: str) -> str:
    # Minimal extraction; good enough for local HTML files.
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", html)
    html = re.sub(r"(?i)<br\\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(p|div|li|tr|h\\d|blockquote)>", "\n", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = re.sub(r"[ \t]{2,}", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise FileReadError(f"PDF support requires 'pypdf'. ({e})") from e

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise FileReadError(f"Failed to open PDF: {path} ({e})") from e

    parts: list[str] = []
    max_pages = int(os.getenv("AGENT_MAX_PDF_PAGES", "60"))
    for i, page in enumerate(reader.pages):
        if max_pages > 0 and i >= max_pages:
            break
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()


def _read_docx(path: Path) -> str:
    def _text_fallback() -> str | None:
        try:
            data = _read_bytes(path, max_bytes=int(os.getenv("AGENT_MAX_FILE_BYTES", "25000000")))
            if _is_probably_binary(data):
                return None
            text = _decode_text_bytes(data).strip()
            return text or None
        except Exception:
            return None

    try:
        from docx import Document  # type: ignore
    except Exception as e:  # pragma: no cover
        fallback = _text_fallback()
        if fallback is not None:
            return fallback
        raise FileReadError(f"DOCX support requires 'python-docx'. ({e})") from e

    try:
        doc = Document(str(path))
    except Exception as e:
        # Some users rename plain text files to .docx by mistake.
        # Fallback: if the content is text-like, decode and continue as plain text.
        fallback = _text_fallback()
        if fallback is not None:
            return fallback
        raise FileReadError(f"Failed to open DOCX: {path} ({e})") from e

    paras = [p.text for p in doc.paragraphs if (p.text or "").strip()]
    return "\n".join(paras).strip()


def _read_pptx(path: Path) -> str:
    try:
        from pptx import Presentation  # type: ignore
    except Exception as e:  # pragma: no cover
        raise FileReadError(f"PPTX support requires 'python-pptx'. ({e})") from e

    try:
        pres = Presentation(str(path))
    except Exception as e:
        raise FileReadError(f"Failed to open PPTX: {path} ({e})") from e

    out: list[str] = []
    max_slides = int(os.getenv("AGENT_MAX_PPTX_SLIDES", "80"))
    for i, slide in enumerate(pres.slides):
        if max_slides > 0 and i >= max_slides:
            break
        out.append(f"[Slide {i+1}]")
        for shape in slide.shapes:
            text = getattr(shape, "text", "") or ""
            text = text.strip()
            if text:
                out.append(text)
        out.append("")
    return "\n".join(out).strip()


def _read_xlsx(path: Path) -> str:
    try:
        import openpyxl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise FileReadError(f"XLSX support requires 'openpyxl'. ({e})") from e

    try:
        wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    except Exception as e:
        raise FileReadError(f"Failed to open XLSX: {path} ({e})") from e

    max_sheets = int(os.getenv("AGENT_MAX_XLSX_SHEETS", "6"))
    max_rows = int(os.getenv("AGENT_MAX_XLSX_ROWS", "200"))
    max_cols = int(os.getenv("AGENT_MAX_XLSX_COLS", "30"))

    out: list[str] = []
    for si, name in enumerate(wb.sheetnames):
        if max_sheets > 0 and si >= max_sheets:
            break
        ws = wb[name]
        out.append(f"[Sheet] {name}")
        for ri, row in enumerate(ws.iter_rows(values_only=True)):
            if max_rows > 0 and ri >= max_rows:
                break
            cells = []
            for ci, val in enumerate(row):
                if max_cols > 0 and ci >= max_cols:
                    break
                if val is None:
                    cells.append("")
                else:
                    cells.append(str(val))
            if any(c.strip() for c in cells):
                out.append("\t".join(cells).rstrip())
        out.append("")
    return "\n".join(out).strip()


def _read_image_ocr(path: Path) -> str:
    data = _read_bytes(path, max_bytes=int(os.getenv("OCR_SPACE_MAX_BYTES", "8000000")))
    lang = os.getenv("OCR_SPACE_LANGUAGE") or os.getenv("OCR_SPACE_LANG")
    try:
        return ocr_api.ocr_image_bytes(data, filename=path.name, language=lang)
    except ocr_api.OCRSpaceError as e:
        raise FileReadError(str(e)) from e


def read_any_file(path: str, *, max_chars: int | None = None) -> str:
    """
    Best-effort text extraction from many file types.

    Supported (when deps are installed): pdf, docx, pptx, xlsx, images (OCR).
    Falls back to decoding as text for unknown extensions.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileReadError(f"File not found: {path}")

    max_chars = int(os.getenv("AGENT_MAX_FILE_CHARS", "200000")) if max_chars is None else int(max_chars)
    max_bytes = int(os.getenv("AGENT_MAX_FILE_BYTES", "25000000"))

    ext = p.suffix.lower()
    if ext == ".pdf":
        return _truncate(_read_pdf(p), max_chars=max_chars)
    if ext == ".docx":
        return _truncate(_read_docx(p), max_chars=max_chars)
    if ext == ".pptx":
        return _truncate(_read_pptx(p), max_chars=max_chars)
    if ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return _truncate(_read_xlsx(p), max_chars=max_chars)
    if ext in {".html", ".htm"}:
        data = _read_bytes(p, max_bytes=max_bytes)
        if _is_probably_binary(data):
            raise FileReadError("HTML file looks binary/unreadable.")
        return _truncate(_html_to_text(_decode_text_bytes(data)), max_chars=max_chars)
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return _truncate(_read_image_ocr(p), max_chars=max_chars)

    data = _read_bytes(p, max_bytes=max_bytes)
    if _is_probably_binary(data):
        raise FileReadError(f"Unsupported binary file type: {ext or '(no extension)'}")
    return _truncate(_decode_text_bytes(data), max_chars=max_chars)
