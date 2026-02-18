from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable


class FileWriteError(RuntimeError):
    pass


def _ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise FileWriteError(f"Failed to create output directory: {path.parent} ({e})") from e


def write_text_file(path: str | Path, text: str) -> None:
    p = Path(path)
    _ensure_parent_dir(p)
    try:
        p.write_text(text, encoding="utf-8")
    except OSError as e:
        raise FileWriteError(f"Failed to write file: {p} ({e})") from e


def write_docx_file(path: str | Path, text: str) -> None:
    try:
        from docx import Document  # type: ignore
    except Exception as e:  # pragma: no cover
        raise FileWriteError(f"DOCX output requires 'python-docx'. ({e})") from e

    p = Path(path)
    _ensure_parent_dir(p)

    doc = Document()
    # Preserve line breaks by mapping each line to a paragraph.
    # This does not preserve the original DOCX formatting (styles/runs/tables).
    for line in (text or "").splitlines():
        doc.add_paragraph(line)

    try:
        doc.save(str(p))
    except OSError as e:
        raise FileWriteError(f"Failed to write DOCX: {p} ({e})") from e


def _add_paragraph_once(paragraph: Any, out: list[Any], seen: set[int]) -> None:
    key = id(paragraph._p)
    if key in seen:
        return
    seen.add(key)
    out.append(paragraph)


def _collect_table_paragraphs(table: Any, out: list[Any], seen: set[int]) -> None:
    for row in table.rows:
        for cell in row.cells:
            _collect_cell_paragraphs(cell, out, seen)


def _collect_cell_paragraphs(cell: Any, out: list[Any], seen: set[int]) -> None:
    for paragraph in cell.paragraphs:
        _add_paragraph_once(paragraph, out, seen)
    for table in cell.tables:
        _collect_table_paragraphs(table, out, seen)


def _collect_docx_paragraphs(doc: Any) -> list[Any]:
    out: list[Any] = []
    seen: set[int] = set()

    for paragraph in doc.paragraphs:
        _add_paragraph_once(paragraph, out, seen)
    for table in doc.tables:
        _collect_table_paragraphs(table, out, seen)

    for section in doc.sections:
        for attr in (
            "header",
            "first_page_header",
            "even_page_header",
            "footer",
            "first_page_footer",
            "even_page_footer",
        ):
            part = getattr(section, attr, None)
            if part is None:
                continue
            for paragraph in part.paragraphs:
                _add_paragraph_once(paragraph, out, seen)
            for table in part.tables:
                _collect_table_paragraphs(table, out, seen)

    return out


def _set_paragraph_text_preserving_runs(paragraph: Any, new_text: str) -> None:
    runs = list(paragraph.runs)
    if not runs:
        paragraph.add_run(new_text)
        return

    old_lengths = [len(run.text or "") for run in runs]
    old_total = sum(old_lengths)
    new_len = len(new_text)

    if new_len == 0:
        for run in runs:
            run.text = ""
        return

    if old_total <= 0:
        runs[0].text = new_text
        for run in runs[1:]:
            run.text = ""
        return

    cuts: list[int] = [0]
    cumulative = 0
    for length in old_lengths[:-1]:
        cumulative += length
        cut = round((cumulative / old_total) * new_len)
        if cut < cuts[-1]:
            cut = cuts[-1]
        if cut > new_len:
            cut = new_len
        cuts.append(cut)
    cuts.append(new_len)

    for idx, run in enumerate(runs):
        run.text = new_text[cuts[idx] : cuts[idx + 1]]


def _should_transform_default(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", s))


def transform_docx_preserving_format(
    input_path: str | Path,
    output_path: str | Path,
    *,
    transform: Callable[[str], str],
    should_transform: Callable[[str], bool] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    try:
        from docx import Document  # type: ignore
    except Exception as e:  # pragma: no cover
        raise FileWriteError(f"DOCX support requires 'python-docx'. ({e})") from e

    src = Path(input_path)
    dst = Path(output_path)
    if not src.exists() or not src.is_file():
        raise FileWriteError(f"Input DOCX not found: {src}")
    _ensure_parent_dir(dst)

    try:
        doc = Document(str(src))
    except Exception as e:
        raise FileWriteError(f"Failed to open DOCX: {src} ({e})") from e

    paragraphs = _collect_docx_paragraphs(doc)
    predicate = should_transform or _should_transform_default
    processed = 0
    changed = 0
    total = len(paragraphs)

    for paragraph in paragraphs:
        processed += 1
        if on_progress is not None:
            on_progress(processed, total)
        original = paragraph.text or ""
        if not predicate(original):
            continue
        updated = transform(original)
        if updated != original:
            _set_paragraph_text_preserving_runs(paragraph, updated)
            changed += 1

    try:
        doc.save(str(dst))
    except OSError as e:
        raise FileWriteError(f"Failed to write DOCX: {dst} ({e})") from e

    return processed, changed


def write_any_file(path: str | Path, text: str) -> None:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".docx":
        write_docx_file(p, text)
        return
    write_text_file(p, text)
