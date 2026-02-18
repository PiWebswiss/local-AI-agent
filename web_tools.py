from __future__ import annotations

import json
import io
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

import ocr_api


class WebToolError(RuntimeError):
    pass


def ensure_https_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise WebToolError("URL is empty.")

    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        url = "https://" + url.lstrip("/")
        parsed = urllib.parse.urlparse(url)

    if parsed.scheme == "https":
        return url

    if parsed.scheme == "http":
        return urllib.parse.urlunparse(parsed._replace(scheme="https"))

    raise WebToolError("Only https URLs are allowed.")


def _normalize_content_source_url(url: str) -> str:
    """
    Normalize source URLs for better content extraction.

    Example: GitHub blob URLs are mapped to raw.githubusercontent.com so
    code files can be fetched as plain text.
    """
    url = ensure_https_url(url)
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return url

    host = (parsed.netloc or "").lower()
    parts = [p for p in (parsed.path or "").split("/") if p]
    if host.endswith("github.com") and len(parts) >= 5 and parts[2] == "blob":
        owner, repo, ref = parts[0], parts[1], parts[3]
        tail = "/".join(parts[4:])
        if tail:
            return urllib.parse.urlunparse(
                (
                    "https",
                    "raw.githubusercontent.com",
                    f"/{owner}/{repo}/{ref}/{tail}",
                    "",
                    "",
                    "",
                )
            )
    return url


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        absolute = urllib.parse.urljoin(req.full_url, newurl)
        absolute = ensure_https_url(absolute)
        return super().redirect_request(req, fp, code, msg, headers, absolute)


_OPENER = urllib.request.build_opener(_HTTPSOnlyRedirectHandler())


def _http_get_bytes(url: str, *, timeout_s: float = 20.0, max_bytes: int = 2_000_000) -> tuple[str, bytes]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; LocalAI-Agent/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )

    try:
        with _OPENER.open(req, timeout=timeout_s) as resp:
            content_type = resp.headers.get("Content-Type", "") or ""
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = resp.read(min(64_000, max_bytes - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
    except urllib.error.HTTPError as e:
        raise WebToolError(f"HTTP {e.code} while fetching {url}") from e
    except urllib.error.URLError as e:
        raise WebToolError(f"Network error while fetching {url}: {e.reason}") from e

    raw = b"".join(chunks)
    return content_type.lower(), raw


def _http_get(url: str, *, timeout_s: float = 20.0, max_bytes: int = 2_000_000) -> tuple[str, str]:
    content_type, raw = _http_get_bytes(url, timeout_s=timeout_s, max_bytes=max_bytes)
    encoding = _guess_encoding(content_type, raw)
    text = raw.decode(encoding, errors="replace")
    return content_type, text


def _guess_encoding(content_type: str, raw: bytes) -> str:
    ct = content_type or ""
    match = re.search(r"charset=([^\s;]+)", ct, re.IGNORECASE)
    if match:
        return match.group(1).strip("\"'").lower()
    # Try to sniff from HTML meta
    head = raw[:50_000].decode("utf-8", errors="ignore")
    match = re.search(r"charset\s*=\s*([a-zA-Z0-9_\-]+)", head, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "utf-8"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"p", "div", "br", "hr", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}:
            self._out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in {"p", "div", "li", "tr", "blockquote"}:
            self._out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        s = (data or "").strip()
        if not s:
            return
        self._out.append(s)
        self._out.append(" ")

    def text(self) -> str:
        raw = "".join(self._out)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        return raw.strip()


def fetch_url(url: str, *, timeout_s: float = 20.0, max_chars: int = 8_000) -> str:
    url = _normalize_content_source_url(url)
    content_type, raw = _http_get_bytes(url, timeout_s=timeout_s, max_bytes=12_000_000)

    lowered = (content_type or "").lower()
    url_lower = url.lower()

    text = ""
    if "application/pdf" in lowered or url_lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(io.BytesIO(raw))
            parts = []
            for i, page in enumerate(reader.pages):
                if i >= 60:
                    break
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    parts.append("")
            text = "\n\n".join([p.strip() for p in parts if p and p.strip()]).strip()
        except Exception as e:
            raise WebToolError(f"Failed to extract PDF text from {url}: {e}") from e
    elif (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in lowered
        or url_lower.endswith(".docx")
    ):
        try:
            from docx import Document  # type: ignore

            doc = Document(io.BytesIO(raw))
            paras = [p.text for p in doc.paragraphs if (p.text or "").strip()]
            text = "\n".join(paras).strip()
        except Exception as e:
            raise WebToolError(f"Failed to extract DOCX text from {url}: {e}") from e
    elif (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" in lowered
        or url_lower.endswith(".pptx")
    ):
        try:
            from pptx import Presentation  # type: ignore

            pres = Presentation(io.BytesIO(raw))
            out: list[str] = []
            for si, slide in enumerate(pres.slides):
                if si >= 80:
                    break
                out.append(f"[Slide {si+1}]")
                for shape in slide.shapes:
                    s = (getattr(shape, "text", "") or "").strip()
                    if s:
                        out.append(s)
                out.append("")
            text = "\n".join(out).strip()
        except Exception as e:
            raise WebToolError(f"Failed to extract PPTX text from {url}: {e}") from e
    elif (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in lowered
        or url_lower.endswith(".xlsx")
        or url_lower.endswith(".xlsm")
    ):
        try:
            import openpyxl  # type: ignore

            wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True, read_only=True)
            out: list[str] = []
            for wi, name in enumerate(wb.sheetnames):
                if wi >= 6:
                    break
                ws = wb[name]
                out.append(f"[Sheet] {name}")
                for ri, row in enumerate(ws.iter_rows(values_only=True)):
                    if ri >= 200:
                        break
                    cells = ["" if v is None else str(v) for v in row[:30]]
                    if any(c.strip() for c in cells):
                        out.append("\t".join(cells).rstrip())
                out.append("")
            text = "\n".join(out).strip()
        except Exception as e:
            raise WebToolError(f"Failed to extract XLSX text from {url}: {e}") from e
    elif lowered.startswith("image/") or any(
        url_lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    ):
        try:
            lang = os.getenv("OCR_SPACE_LANGUAGE") or os.getenv("OCR_SPACE_LANG")
            path = urllib.parse.urlparse(url).path or ""
            fname = (path.rsplit("/", 1)[-1] or "image").strip() or "image"
            text = ocr_api.ocr_image_bytes(raw, filename=fname, language=lang)
        except Exception as e:
            raise WebToolError(f"Failed to OCR image from {url}: {e}") from e
    else:
        encoding = _guess_encoding(content_type, raw)
        text = raw.decode(encoding, errors="replace")
        if "text/html" in lowered or "<html" in text[:500].lower():
            parser = _HTMLTextExtractor()
            parser.feed(text)
            text = parser.text()
        else:
            text = text.strip()

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def _decode_ddg_href(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/duckduckgo.com/"):
        href = "https://" + href.lstrip("/")
    if href.startswith("/"):
        href = "https://duckduckgo.com" + href
    try:
        parsed = urllib.parse.urlparse(href)
        path = (parsed.path or "").rstrip("/")
        last_seg = path.rsplit("/", 1)[-1] if path else ""
        if parsed.netloc.endswith("duckduckgo.com") and last_seg == "l":
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return urllib.parse.unquote(qs["uddg"][0])
    except Exception:
        return href
    return href


class _DuckDuckGoHTMLResults(HTMLParser):
    def __init__(self, *, max_results: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_results = max_results
        self.results: list[dict[str, str]] = []
        self._capturing_title = False
        self._current: dict[str, str] | None = None
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self.results) >= self.max_results:
            return
        if tag.lower() != "a":
            return
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        cls = attr_map.get("class", "")
        href = attr_map.get("href", "")
        if not href:
            return
        if "result__a" in cls or "result-link" in cls:
            url = _decode_ddg_href(href)
            self._current = {"title": "", "url": url, "snippet": ""}
            self._title_parts = []
            self._capturing_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a":
            return
        if self._capturing_title and self._current is not None:
            title = " ".join([p.strip() for p in self._title_parts if p.strip()]).strip()
            title = re.sub(r"\s{2,}", " ", title)
            self._current["title"] = title
            if self._current.get("url") and self._current.get("title"):
                self.results.append(self._current)
            self._current = None
            self._title_parts = []
            self._capturing_title = False

    def handle_data(self, data: str) -> None:
        if not self._capturing_title:
            return
        if self._current is None:
            return
        s = (data or "").strip()
        if s:
            self._title_parts.append(s)


class _DuckDuckGoLiteResults(HTMLParser):
    def __init__(self, *, max_results: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_results = max_results
        self.results: list[dict[str, str]] = []
        self._capturing_title = False
        self._current: dict[str, str] | None = None
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self.results) >= self.max_results:
            return
        if tag.lower() != "a":
            return
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        href = attr_map.get("href", "")
        if not href:
            return

        url = _decode_ddg_href(href)
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            return
        if parsed.scheme not in {"http", "https"}:
            return
        if not parsed.netloc:
            return
        if parsed.netloc.endswith("duckduckgo.com"):
            return

        self._current = {"title": "", "url": url, "snippet": ""}
        self._title_parts = []
        self._capturing_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a":
            return
        if self._capturing_title and self._current is not None:
            title = " ".join([p.strip() for p in self._title_parts if p.strip()]).strip()
            title = re.sub(r"\s{2,}", " ", title)
            self._current["title"] = title
            if self._current.get("url") and self._current.get("title"):
                self.results.append(self._current)
            self._current = None
            self._title_parts = []
            self._capturing_title = False

    def handle_data(self, data: str) -> None:
        if not self._capturing_title:
            return
        if self._current is None:
            return
        s = (data or "").strip()
        if s:
            self._title_parts.append(s)


def _duckduckgo_html_search(query: str, *, max_results: int) -> list[dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={q}"
    content_type, page = _http_get(url, timeout_s=20.0, max_bytes=1_500_000)
    if "text/html" not in content_type and "<html" not in page[:300].lower():
        raise WebToolError("DuckDuckGo did not return HTML.")
    parser = _DuckDuckGoHTMLResults(max_results=max_results)
    parser.feed(page)
    return parser.results


def _duckduckgo_lite_search(query: str, *, max_results: int) -> list[dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={q}"
    content_type, page = _http_get(url, timeout_s=20.0, max_bytes=1_500_000)
    if "text/html" not in content_type and "<html" not in page[:300].lower():
        raise WebToolError("DuckDuckGo Lite did not return HTML.")
    parser = _DuckDuckGoLiteResults(max_results=max_results)
    parser.feed(page)
    return parser.results


def _duckduckgo_instant_answer(query: str, *, max_results: int) -> list[dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
    _, text = _http_get(url, timeout_s=20.0, max_bytes=2_000_000)
    data = json.loads(text)
    out: list[dict[str, str]] = []

    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    if abstract_url:
        out.append({"title": (data.get("Heading") or "DuckDuckGo Abstract").strip(), "url": abstract_url, "snippet": abstract})

    def add_topic(item: dict[str, Any]) -> None:
        nonlocal out
        url2 = (item.get("FirstURL") or "").strip()
        txt = (item.get("Text") or "").strip()
        if url2 and txt:
            out.append({"title": txt, "url": url2, "snippet": txt})

    topics = data.get("RelatedTopics") or []
    for item in topics:
        if len(out) >= max_results:
            break
        if isinstance(item, dict) and "Topics" in item and isinstance(item["Topics"], list):
            for sub in item["Topics"]:
                if len(out) >= max_results:
                    break
                if isinstance(sub, dict):
                    add_topic(sub)
        elif isinstance(item, dict):
            add_topic(item)

    return out[:max_results]


def web_search(query: str, *, max_results: int = 5) -> list[dict[str, str]]:
    query = (query or "").strip()
    if not query:
        raise WebToolError("Query is empty.")
    max_results = max(1, min(int(max_results), 10))

    def normalize_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in results:
            url = (item.get("url") or "").strip()
            if not url:
                continue
            url = _decode_ddg_href(url)
            try:
                url = ensure_https_url(url)
            except WebToolError:
                continue
            try:
                parsed = urllib.parse.urlparse(url)
                if parsed.netloc.endswith("duckduckgo.com"):
                    continue
            except Exception:
                continue
            if url in seen:
                continue
            seen.add(url)
            out.append(
                {
                    "title": (item.get("title") or "").strip(),
                    "url": url,
                    "snippet": (item.get("snippet") or "").strip(),
                }
            )
            if len(out) >= max_results:
                break
        return out

    results: list[dict[str, str]] = []
    try:
        results = normalize_results(_duckduckgo_html_search(query, max_results=max_results))
    except Exception:
        results = []

    if results:
        return results[:max_results]

    try:
        results = normalize_results(_duckduckgo_lite_search(query, max_results=max_results))
    except Exception:
        results = []

    if results:
        return results[:max_results]

    try:
        results = normalize_results(_duckduckgo_instant_answer(query, max_results=max_results))
        return results[:max_results]
    except Exception as e:
        raise WebToolError(f"Search failed: {e}") from e
