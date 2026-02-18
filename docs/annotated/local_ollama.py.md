# local_ollama.py — line-by-line explanation

Generated: 2026-02-18T09:56:40

Purpose: Minimal Ollama HTTP client + Gemma3 1B/4B model validation.

Format: each original source line is shown with its line number, followed by a short explanation.

```text
  1 from __future__ import annotations
      ↳ Imports annotations from the third-party module `__future__`.
  2 
      ↳ Blank line for readability.
  3 import json
      ↳ Imports standard library modules: json.
  4 import os
      ↳ Imports standard library modules: os.
  5 import re
      ↳ Imports standard library modules: re.
  6 import urllib.error
      ↳ Imports standard library modules: urllib.error.
  7 import urllib.parse
      ↳ Imports standard library modules: urllib.parse.
  8 import urllib.request
      ↳ Imports standard library modules: urllib.request.
  9 from dataclasses import dataclass
      ↳ Imports dataclass from the standard library module `dataclasses`.
 10 from typing import Any
      ↳ Imports Any from the standard library module `typing`.
 11 
      ↳ Blank line for readability.
 12 
      ↳ Blank line for readability.
 13 class OllamaError(RuntimeError):
      ↳ Defines a custom exception class `OllamaError`.
 14     pass
      ↳ Control-flow keyword.
 15 
      ↳ Blank line for readability.
 16 
      ↳ Blank line for readability.
 17 _GEMMA3_ALLOWED_TAG_RE = re.compile(r"^gemma3:(?P<size>[0-9]+)b(?P<suffix>[-_].+)?$", re.IGNORECASE)
      ↳ Assignment: sets `_GEMMA3_ALLOWED_TAG_RE`.
 18 
      ↳ Blank line for readability.
 19 
      ↳ Blank line for readability.
 20 def validate_gemma3_model(model: str, *, max_b: int = 4) -> str:
      ↳ Defines `validate_gemma3_model()`: Allow only `gemma3:1b` or `gemma3:4b` (with optional suffix).
 21     model = (model or "").strip()
      ↳ Assignment: sets `model`.
 22     match = _GEMMA3_ALLOWED_TAG_RE.match(model)
      ↳ Assignment: sets `match`.
 23     if not match:
      ↳ Conditional branch: checks a condition and chooses a code path.
 24         raise OllamaError(f"Unsupported model '{model}'. Use 'gemma3:1b' or 'gemma3:4b'.")
      ↳ Raises an exception to signal an error.
 25     size_b = int(match.group("size"))
      ↳ Assignment: sets `size_b`.
 26     if size_b > max_b:
      ↳ Conditional branch: checks a condition and chooses a code path.
 27         raise OllamaError(f"Model '{model}' exceeds max size ({max_b}b).")
      ↳ Raises an exception to signal an error.
 28     if size_b not in (1, 4):
      ↳ Conditional branch: checks a condition and chooses a code path.
 29         raise OllamaError(f"Unsupported gemma3 size '{size_b}b'. Use 1b or 4b.")
      ↳ Raises an exception to signal an error.
 30     return model
      ↳ Returns a value from the current function.
 31 
      ↳ Blank line for readability.
 32 
      ↳ Blank line for readability.
 33 def _normalize_host(host: str) -> str:
      ↳ Defines `_normalize_host()`: Normalize Ollama host string (scheme + no trailing slash).
 34     host = (host or "").strip()
      ↳ Assignment: sets `host`.
 35     if not host:
      ↳ Conditional branch: checks a condition and chooses a code path.
 36         return "http://localhost:11434"
      ↳ Returns a value from the current function.
 37     if not host.startswith(("http://", "https://")):
      ↳ Conditional branch: checks a condition and chooses a code path.
 38         host = "http://" + host
      ↳ Assignment: sets `host`.
 39     return host.rstrip("/")
      ↳ Returns a value from the current function.
 40 
      ↳ Blank line for readability.
 41 
      ↳ Blank line for readability.
 42 @dataclass(frozen=True)
      ↳ Decorator line: modifies the behavior of the next function/method.
 43 class OllamaConfig:
      ↳ Defines a class `OllamaConfig`.
 44     host: str
      ↳ Implementation detail: part of the surrounding logic.
 45     model: str
      ↳ Implementation detail: part of the surrounding logic.
 46     timeout_s: float = 120.0
      ↳ Assignment: sets `timeout_s: float`.
 47 
      ↳ Blank line for readability.
 48     @staticmethod
      ↳ Decorator line: modifies the behavior of the next function/method.
 49     def from_env() -> "OllamaConfig":
      ↳ Defines function `from_env()`.
 50         host = _normalize_host(os.getenv("OLLAMA_HOST", "http://localhost:11434"))
      ↳ Assignment: sets `host`.
 51         model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
      ↳ Assignment: sets `model`.
 52         model = validate_gemma3_model(model, max_b=int(os.getenv("OLLAMA_MAX_B", "4")))
      ↳ Assignment: sets `model`.
 53         timeout_s = float(os.getenv("OLLAMA_TIMEOUT_S", "120"))
      ↳ Assignment: sets `timeout_s`.
 54         return OllamaConfig(host=host, model=model, timeout_s=timeout_s)
      ↳ Returns a value from the current function.
 55 
      ↳ Blank line for readability.
 56 
      ↳ Blank line for readability.
 57 def chat(
      ↳ Defines `chat()`: Call Ollama's `/api/chat` endpoint (non-streaming).
 58     *,
      ↳ Implementation detail: part of the surrounding logic.
 59     config: OllamaConfig,
      ↳ Implementation detail: part of the surrounding logic.
 60     messages: list[dict[str, str]],
      ↳ Implementation detail: part of the surrounding logic.
 61     options: dict[str, Any] | None = None,
      ↳ Assignment: sets `options: dict[str, Any] | None`.
 62 ) -> str:
      ↳ Starts a new block (indented section) in Python.
 63     url = f"{config.host}/api/chat"
      ↳ Assignment: sets `url`.
 64     payload: dict[str, Any] = {
      ↳ Assignment: sets `payload: dict[str, Any]`.
 65         "model": config.model,
      ↳ Implementation detail: part of the surrounding logic.
 66         "messages": messages,
      ↳ Implementation detail: part of the surrounding logic.
 67         "stream": False,
      ↳ Implementation detail: part of the surrounding logic.
 68     }
      ↳ Implementation detail: part of the surrounding logic.
 69     if options:
      ↳ Conditional branch: checks a condition and chooses a code path.
 70         payload["options"] = options
      ↳ Assignment: sets `payload["options"]`.
 71 
      ↳ Blank line for readability.
 72     req = urllib.request.Request(
      ↳ Assignment: sets `req`.
 73         url,
      ↳ Implementation detail: part of the surrounding logic.
 74         method="POST",
      ↳ Assignment: sets `method`.
 75         data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
      ↳ Assignment: sets `data`.
 76         headers={
      ↳ Assignment: sets `headers`.
 77             "Content-Type": "application/json; charset=utf-8",
      ↳ Assignment: sets `"Content-Type": "application/json; charset`.
 78             "Accept": "application/json",
      ↳ Implementation detail: part of the surrounding logic.
 79         },
      ↳ Implementation detail: part of the surrounding logic.
 80     )
      ↳ Implementation detail: part of the surrounding logic.
 81 
      ↳ Blank line for readability.
 82     try:
      ↳ Start of a `try` block for exception handling.
 83         with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
      ↳ Context manager block: ensures setup/teardown around a resource.
 84             data = resp.read()
      ↳ Assignment: sets `data`.
 85     except urllib.error.HTTPError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
 86         body = ""
      ↳ Assignment: sets `body`.
 87         try:
      ↳ Start of a `try` block for exception handling.
 88             body = e.read().decode("utf-8", errors="replace")
      ↳ Assignment: sets `body`.
 89         except Exception:
      ↳ Exception handler: runs if the `try` block raises an error.
 90             body = ""
      ↳ Assignment: sets `body`.
 91         raise OllamaError(f"Ollama HTTP {e.code}: {body or e.reason}") from e
      ↳ Raises an exception to signal an error.
 92     except urllib.error.URLError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
 93         raise OllamaError(
      ↳ Raises an exception to signal an error.
 94             f"Failed to reach Ollama at {config.host}. "
      ↳ Implementation detail: part of the surrounding logic.
 95             f"Is Ollama running and accessible? ({e.reason})"
      ↳ Implementation detail: part of the surrounding logic.
 96         ) from e
      ↳ Implementation detail: part of the surrounding logic.
 97 
      ↳ Blank line for readability.
 98     try:
      ↳ Start of a `try` block for exception handling.
 99         obj = json.loads(data.decode("utf-8", errors="replace"))
      ↳ Assignment: sets `obj`.
100     except json.JSONDecodeError as e:
      ↳ Exception handler: runs if the `try` block raises an error.
101         raise OllamaError(f"Invalid JSON from Ollama: {e}") from e
      ↳ Raises an exception to signal an error.
102 
      ↳ Blank line for readability.
103     content = (obj.get("message") or {}).get("content") or ""
      ↳ Assignment: sets `content`.
104     if not isinstance(content, str):
      ↳ Conditional branch: checks a condition and chooses a code path.
105         raise OllamaError("Unexpected response shape from Ollama.")
      ↳ Raises an exception to signal an error.
106     return content.strip()
      ↳ Returns a value from the current function.
107 
      ↳ Blank line for readability.
```
