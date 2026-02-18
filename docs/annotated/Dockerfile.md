# Dockerfile — line-by-line explanation

Generated: 2026-02-18T09:56:40

Purpose: Builds the agent container image (Python slim + deps, run as non-root).

Format: each original source line is shown with its line number, followed by a short explanation.

```text
  1 FROM python:3.12-slim
      ↳ Implementation detail: part of the surrounding logic.
  2 
      ↳ Blank line for readability.
  3 WORKDIR /app
      ↳ Implementation detail: part of the surrounding logic.
  4 
      ↳ Blank line for readability.
  5 ENV PYTHONDONTWRITEBYTECODE=1 \
      ↳ Assignment: sets `ENV PYTHONDONTWRITEBYTECODE`.
  6     PYTHONUNBUFFERED=1
      ↳ Assignment: sets `PYTHONUNBUFFERED`.
  7 
      ↳ Blank line for readability.
  8 RUN apt-get update \
      ↳ Implementation detail: part of the surrounding logic.
  9     && apt-get install -y --no-install-recommends ca-certificates \
      ↳ Implementation detail: part of the surrounding logic.
 10     && rm -rf /var/lib/apt/lists/*
      ↳ Implementation detail: part of the surrounding logic.
 11 
      ↳ Blank line for readability.
 12 RUN useradd --create-home --uid 10001 --shell /bin/false appuser
      ↳ Implementation detail: part of the surrounding logic.
 13 
      ↳ Blank line for readability.
 14 COPY requirements.txt /app/requirements.txt
      ↳ Implementation detail: part of the surrounding logic.
 15 RUN pip install --no-cache-dir -r /app/requirements.txt
      ↳ Implementation detail: part of the surrounding logic.
 16 
      ↳ Blank line for readability.
 17 COPY . /app
      ↳ Implementation detail: part of the surrounding logic.
 18 
      ↳ Blank line for readability.
 19 USER appuser
      ↳ Implementation detail: part of the surrounding logic.
 20 CMD ["python", "agent.py", "chat"]
      ↳ Implementation detail: part of the surrounding logic.
```
