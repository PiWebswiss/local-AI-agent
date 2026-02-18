# file_writers.py — line-by-line explanation

Generated: 2026-02-18T09:56:40

Purpose: Safe local output writing helpers (writes text or DOCX to the output directory).

Format: each original source line is shown with its line number, followed by a short explanation.

```text
  1 from __future__ import annotations
      ↳ Imports annotations from the third-party module `__future__`.
  2 
      ↳ Blank line for readability.
  3 import re
      ↳ Imports standard library modules: re.
  4 from pathlib import Path
      ↳ Imports Path from the standard library module `pathlib`.
  5 from typing import Any, Callable
      ↳ Imports Any, Callable from the standard library module `typing`.
  6 
      ↳ Blank line for readability.
  7 
      ↳ Blank line for readability.
  8 class FileWriteError(RuntimeError):
      ↳ Defines a custom exception class `FileWriteError`.
  9     pass
      ↳ Control-flow keyword.
 10 
      ↳ Blank line for readability.
 11 
      ↳ Blank line for readability.
 12 def _ensure_parent_dir(path: Path) -> None:
      ↳ Defines function `_ensure_parent_dir()`.
 13     try:
      ↳ Start of a `try` block for exception handling.
 14         path.parent.mkdir(parents=True, exist_ok=True)
      ↳ Assignment: sets `path.parent.mkdir(parents`.
 15     except OSError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
 16         raise FileWriteError(f"Failed to create output directory: {path.parent} ({e})") from e
      ↳ Raises an exception to signal an error.
 17 
      ↳ Blank line for readability.
 18 
      ↳ Blank line for readability.
 19 def write_text_file(path: str | Path, text: str) -> None:
      ↳ Defines `write_text_file()`: Write UTF-8 text to a local file (creating parent directories).
 20     p = Path(path)
      ↳ Assignment: sets `p`.
 21     _ensure_parent_dir(p)
      ↳ Implementation detail: part of the surrounding logic.
 22     try:
      ↳ Start of a `try` block for exception handling.
 23         p.write_text(text, encoding="utf-8")
      ↳ Assignment: sets `p.write_text(text, encoding`.
 24     except OSError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
 25         raise FileWriteError(f"Failed to write file: {p} ({e})") from e
      ↳ Raises an exception to signal an error.
 26 
      ↳ Blank line for readability.
 27 
      ↳ Blank line for readability.
 28 def write_docx_file(path: str | Path, text: str) -> None:
      ↳ Defines `write_docx_file()`: Write text into a new DOCX (one paragraph per line).
 29     try:
      ↳ Start of a `try` block for exception handling.
 30         from docx import Document  # type: ignore
      ↳ Lazy/inner-scope imports Document  # type: ignore from the third-party module `docx`.
 31     except Exception as e:  # pragma: no cover
      ↳ Exception handler: runs if the `try` block raises an error.
 32         raise FileWriteError(f"DOCX output requires 'python-docx'. ({e})") from e
      ↳ Raises an exception to signal an error.
 33 
      ↳ Blank line for readability.
 34     p = Path(path)
      ↳ Assignment: sets `p`.
 35     _ensure_parent_dir(p)
      ↳ Implementation detail: part of the surrounding logic.
 36 
      ↳ Blank line for readability.
 37     doc = Document()
      ↳ Assignment: sets `doc`.
 38     # Preserve line breaks by mapping each line to a paragraph.
      ↳ Comment/documentation line.
 39     # This does not preserve the original DOCX formatting (styles/runs/tables).
      ↳ Comment/documentation line.
 40     for line in (text or "").splitlines():
      ↳ Loop: repeats the following block.
 41         doc.add_paragraph(line)
      ↳ Implementation detail: part of the surrounding logic.
 42 
      ↳ Blank line for readability.
 43     try:
      ↳ Start of a `try` block for exception handling.
 44         doc.save(str(p))
      ↳ Implementation detail: part of the surrounding logic.
 45     except OSError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
 46         raise FileWriteError(f"Failed to write DOCX: {p} ({e})") from e
      ↳ Raises an exception to signal an error.
 47 
      ↳ Blank line for readability.
 48 
      ↳ Blank line for readability.
 49 def _add_paragraph_once(paragraph: Any, out: list[Any], seen: set[int]) -> None:
      ↳ Defines function `_add_paragraph_once()`.
 50     key = id(paragraph._p)
      ↳ Assignment: sets `key`.
 51     if key in seen:
      ↳ Conditional branch: checks a condition and chooses a code path.
 52         return
      ↳ Returns a value from the current function.
 53     seen.add(key)
      ↳ Implementation detail: part of the surrounding logic.
 54     out.append(paragraph)
      ↳ Implementation detail: part of the surrounding logic.
 55 
      ↳ Blank line for readability.
 56 
      ↳ Blank line for readability.
 57 def _collect_table_paragraphs(table: Any, out: list[Any], seen: set[int]) -> None:
      ↳ Defines function `_collect_table_paragraphs()`.
 58     for row in table.rows:
      ↳ Loop: repeats the following block.
 59         for cell in row.cells:
      ↳ Loop: repeats the following block.
 60             _collect_cell_paragraphs(cell, out, seen)
      ↳ Implementation detail: part of the surrounding logic.
 61 
      ↳ Blank line for readability.
 62 
      ↳ Blank line for readability.
 63 def _collect_cell_paragraphs(cell: Any, out: list[Any], seen: set[int]) -> None:
      ↳ Defines function `_collect_cell_paragraphs()`.
 64     for paragraph in cell.paragraphs:
      ↳ Loop: repeats the following block.
 65         _add_paragraph_once(paragraph, out, seen)
      ↳ Implementation detail: part of the surrounding logic.
 66     for table in cell.tables:
      ↳ Loop: repeats the following block.
 67         _collect_table_paragraphs(table, out, seen)
      ↳ Implementation detail: part of the surrounding logic.
 68 
      ↳ Blank line for readability.
 69 
      ↳ Blank line for readability.
 70 def _collect_docx_paragraphs(doc: Any) -> list[Any]:
      ↳ Defines function `_collect_docx_paragraphs()`.
 71     out: list[Any] = []
      ↳ Assignment: sets `out: list[Any]`.
 72     seen: set[int] = set()
      ↳ Assignment: sets `seen: set[int]`.
 73 
      ↳ Blank line for readability.
 74     for paragraph in doc.paragraphs:
      ↳ Loop: repeats the following block.
 75         _add_paragraph_once(paragraph, out, seen)
      ↳ Implementation detail: part of the surrounding logic.
 76     for table in doc.tables:
      ↳ Loop: repeats the following block.
 77         _collect_table_paragraphs(table, out, seen)
      ↳ Implementation detail: part of the surrounding logic.
 78 
      ↳ Blank line for readability.
 79     for section in doc.sections:
      ↳ Loop: repeats the following block.
 80         for attr in (
      ↳ Loop: repeats the following block.
 81             "header",
      ↳ Implementation detail: part of the surrounding logic.
 82             "first_page_header",
      ↳ Implementation detail: part of the surrounding logic.
 83             "even_page_header",
      ↳ Implementation detail: part of the surrounding logic.
 84             "footer",
      ↳ Implementation detail: part of the surrounding logic.
 85             "first_page_footer",
      ↳ Implementation detail: part of the surrounding logic.
 86             "even_page_footer",
      ↳ Implementation detail: part of the surrounding logic.
 87         ):
      ↳ Starts a new block (indented section) in Python.
 88             part = getattr(section, attr, None)
      ↳ Assignment: sets `part`.
 89             if part is None:
      ↳ Conditional branch: checks a condition and chooses a code path.
 90                 continue
      ↳ Control-flow keyword.
 91             for paragraph in part.paragraphs:
      ↳ Loop: repeats the following block.
 92                 _add_paragraph_once(paragraph, out, seen)
      ↳ Implementation detail: part of the surrounding logic.
 93             for table in part.tables:
      ↳ Loop: repeats the following block.
 94                 _collect_table_paragraphs(table, out, seen)
      ↳ Implementation detail: part of the surrounding logic.
 95 
      ↳ Blank line for readability.
 96     return out
      ↳ Returns a value from the current function.
 97 
      ↳ Blank line for readability.
 98 
      ↳ Blank line for readability.
 99 def _set_paragraph_text_preserving_runs(paragraph: Any, new_text: str) -> None:
      ↳ Defines function `_set_paragraph_text_preserving_runs()`.
100     runs = list(paragraph.runs)
      ↳ Assignment: sets `runs`.
101     if not runs:
      ↳ Conditional branch: checks a condition and chooses a code path.
102         paragraph.add_run(new_text)
      ↳ Implementation detail: part of the surrounding logic.
103         return
      ↳ Returns a value from the current function.
104 
      ↳ Blank line for readability.
105     old_lengths = [len(run.text or "") for run in runs]
      ↳ Assignment: sets `old_lengths`.
106     old_total = sum(old_lengths)
      ↳ Assignment: sets `old_total`.
107     new_len = len(new_text)
      ↳ Assignment: sets `new_len`.
108 
      ↳ Blank line for readability.
109     if new_len == 0:
      ↳ Conditional branch: checks a condition and chooses a code path.
110         for run in runs:
      ↳ Loop: repeats the following block.
111             run.text = ""
      ↳ Assignment: sets `run.text`.
112         return
      ↳ Returns a value from the current function.
113 
      ↳ Blank line for readability.
114     if old_total <= 0:
      ↳ Conditional branch: checks a condition and chooses a code path.
115         runs[0].text = new_text
      ↳ Assignment: sets `runs[0].text`.
116         for run in runs[1:]:
      ↳ Loop: repeats the following block.
117             run.text = ""
      ↳ Assignment: sets `run.text`.
118         return
      ↳ Returns a value from the current function.
119 
      ↳ Blank line for readability.
120     cuts: list[int] = [0]
      ↳ Assignment: sets `cuts: list[int]`.
121     cumulative = 0
      ↳ Assignment: sets `cumulative`.
122     for length in old_lengths[:-1]:
      ↳ Loop: repeats the following block.
123         cumulative += length
      ↳ Assignment: sets `cumulative +`.
124         cut = round((cumulative / old_total) * new_len)
      ↳ Assignment: sets `cut`.
125         if cut < cuts[-1]:
      ↳ Conditional branch: checks a condition and chooses a code path.
126             cut = cuts[-1]
      ↳ Assignment: sets `cut`.
127         if cut > new_len:
      ↳ Conditional branch: checks a condition and chooses a code path.
128             cut = new_len
      ↳ Assignment: sets `cut`.
129         cuts.append(cut)
      ↳ Implementation detail: part of the surrounding logic.
130     cuts.append(new_len)
      ↳ Implementation detail: part of the surrounding logic.
131 
      ↳ Blank line for readability.
132     for idx, run in enumerate(runs):
      ↳ Loop: repeats the following block.
133         run.text = new_text[cuts[idx] : cuts[idx + 1]]
      ↳ Assignment: sets `run.text`.
134 
      ↳ Blank line for readability.
135 
      ↳ Blank line for readability.
136 def _should_transform_default(text: str) -> bool:
      ↳ Defines function `_should_transform_default()`.
137     s = (text or "").strip()
      ↳ Assignment: sets `s`.
138     if not s:
      ↳ Conditional branch: checks a condition and chooses a code path.
139         return False
      ↳ Returns a value from the current function.
140     return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", s))
      ↳ Returns a value from the current function.
141 
      ↳ Blank line for readability.
142 
      ↳ Blank line for readability.
143 def transform_docx_preserving_format(
      ↳ Defines function `transform_docx_preserving_format()`.
144     input_path: str | Path,
      ↳ Implementation detail: part of the surrounding logic.
145     output_path: str | Path,
      ↳ Implementation detail: part of the surrounding logic.
146     *,
      ↳ Implementation detail: part of the surrounding logic.
147     transform: Callable[[str], str],
      ↳ Implementation detail: part of the surrounding logic.
148     should_transform: Callable[[str], bool] | None = None,
      ↳ Assignment: sets `should_transform: Callable[[str], bool] | None`.
149     on_progress: Callable[[int, int], None] | None = None,
      ↳ Assignment: sets `on_progress: Callable[[int, int], None] | None`.
150 ) -> tuple[int, int]:
      ↳ Starts a new block (indented section) in Python.
151     try:
      ↳ Start of a `try` block for exception handling.
152         from docx import Document  # type: ignore
      ↳ Lazy/inner-scope imports Document  # type: ignore from the third-party module `docx`.
153     except Exception as e:  # pragma: no cover
      ↳ Exception handler: runs if the `try` block raises an error.
154         raise FileWriteError(f"DOCX support requires 'python-docx'. ({e})") from e
      ↳ Raises an exception to signal an error.
155 
      ↳ Blank line for readability.
156     src = Path(input_path)
      ↳ Assignment: sets `src`.
157     dst = Path(output_path)
      ↳ Assignment: sets `dst`.
158     if not src.exists() or not src.is_file():
      ↳ Conditional branch: checks a condition and chooses a code path.
159         raise FileWriteError(f"Input DOCX not found: {src}")
      ↳ Raises an exception to signal an error.
160     _ensure_parent_dir(dst)
      ↳ Implementation detail: part of the surrounding logic.
161 
      ↳ Blank line for readability.
162     try:
      ↳ Start of a `try` block for exception handling.
163         doc = Document(str(src))
      ↳ Assignment: sets `doc`.
164     except Exception as e:
      ↳ Exception handler: runs if the `try` block raises an error.
165         raise FileWriteError(f"Failed to open DOCX: {src} ({e})") from e
      ↳ Raises an exception to signal an error.
166 
      ↳ Blank line for readability.
167     paragraphs = _collect_docx_paragraphs(doc)
      ↳ Assignment: sets `paragraphs`.
168     predicate = should_transform or _should_transform_default
      ↳ Assignment: sets `predicate`.
169     processed = 0
      ↳ Assignment: sets `processed`.
170     changed = 0
      ↳ Assignment: sets `changed`.
171     total = len(paragraphs)
      ↳ Assignment: sets `total`.
172 
      ↳ Blank line for readability.
173     for paragraph in paragraphs:
      ↳ Loop: repeats the following block.
174         processed += 1
      ↳ Assignment: sets `processed +`.
175         if on_progress is not None:
      ↳ Conditional branch: checks a condition and chooses a code path.
176             on_progress(processed, total)
      ↳ Implementation detail: part of the surrounding logic.
177         original = paragraph.text or ""
      ↳ Assignment: sets `original`.
178         if not predicate(original):
      ↳ Conditional branch: checks a condition and chooses a code path.
179             continue
      ↳ Control-flow keyword.
180         updated = transform(original)
      ↳ Assignment: sets `updated`.
181         if updated != original:
      ↳ Conditional branch: checks a condition and chooses a code path.
182             _set_paragraph_text_preserving_runs(paragraph, updated)
      ↳ Implementation detail: part of the surrounding logic.
183             changed += 1
      ↳ Assignment: sets `changed +`.
184 
      ↳ Blank line for readability.
185     try:
      ↳ Start of a `try` block for exception handling.
186         doc.save(str(dst))
      ↳ Implementation detail: part of the surrounding logic.
187     except OSError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
188         raise FileWriteError(f"Failed to write DOCX: {dst} ({e})") from e
      ↳ Raises an exception to signal an error.
189 
      ↳ Blank line for readability.
190     return processed, changed
      ↳ Returns a value from the current function.
191 
      ↳ Blank line for readability.
192 
      ↳ Blank line for readability.
193 def write_any_file(path: str | Path, text: str) -> None:
      ↳ Defines `write_any_file()`: Choose an output writer based on the target extension.
194     p = Path(path)
      ↳ Assignment: sets `p`.
195     ext = p.suffix.lower()
      ↳ Assignment: sets `ext`.
196     if ext == ".docx":
      ↳ Conditional branch: checks a condition and chooses a code path.
197         write_docx_file(p, text)
      ↳ Implementation detail: part of the surrounding logic.
198         return
      ↳ Returns a value from the current function.
199     write_text_file(p, text)
      ↳ Implementation detail: part of the surrounding logic.
```
