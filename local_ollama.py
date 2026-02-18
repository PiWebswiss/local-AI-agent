# Minimal Ollama client wrapper.
# This module validates model settings and sends chat requests to Ollama's HTTP API.
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    # Raised for model validation, HTTP transport, and response-shape errors.
    pass


# Matches model size suffixes like `1b`, `4b`, `70b`.
_MODEL_SIZE_B_RE = re.compile(r"(?<!\d)(?P<size>[0-9]+)b(?!\w)", re.IGNORECASE)


def validate_model(model: str, *, max_b: int = 4) -> str:
    # Normalize whitespace and reject empty model names.
    model = (model or "").strip()
    if not model:
        raise OllamaError("Model name is empty.")

    # Optional size guardrail: block models above configured `max_b`.
    if max_b > 0:
        match = _MODEL_SIZE_B_RE.search(model)
        if match:
            size_b = int(match.group("size"))
            if size_b > max_b:
                raise OllamaError(f"Model '{model}' exceeds max size ({max_b}b).")
    # Return validated model string unchanged.
    return model


def validate_gemma3_model(model: str, *, max_b: int = 4) -> str:
    # Backward-compatible alias for previous code paths.
    return validate_model(model, max_b=max_b)


def _normalize_host(host: str) -> str:
    # Normalize host and provide a default local endpoint.
    host = (host or "").strip()
    if not host:
        return "http://localhost:11434"
    # Allow values like `localhost:11434` by prepending HTTP scheme.
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    # Remove trailing slash so path joins stay stable.
    return host.rstrip("/")


@dataclass(frozen=True)
class OllamaConfig:
    host: str
    model: str
    timeout_s: float = 120.0

    @staticmethod
    def from_env() -> "OllamaConfig":
        # Read host/model/timeout from environment with safe defaults.
        host = _normalize_host(os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
        # Apply configured model-size cap (0 disables cap).
        model = validate_model(model, max_b=int(os.getenv("OLLAMA_MAX_B", "4")))
        timeout_s = float(os.getenv("OLLAMA_TIMEOUT_S", "120"))
        return OllamaConfig(host=host, model=model, timeout_s=timeout_s)


# Low-level chat call used by all agent answer paths.
def chat(
    *,
    config: OllamaConfig,
    messages: list[dict[str, str]],
    options: dict[str, Any] | None = None,
) -> str:
    # Build Ollama chat endpoint URL.
    url = f"{config.host}/api/chat"
    # Build non-streaming payload; caller expects one final response string.
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": False,
    }
    # Include model options only when provided.
    if options:
        payload["options"] = options

    # Build HTTP POST request with JSON body.
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
    )

    try:
        # Send request and read raw response bytes.
        with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        # Try to include server-provided JSON/text body for easier debugging.
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise OllamaError(f"Ollama HTTP {e.code}: {body or e.reason}") from e
    except urllib.error.URLError as e:
        # Network-level failure (DNS/connection/timeout).
        raise OllamaError(
            f"Failed to reach Ollama at {config.host}. "
            f"Is Ollama running and accessible? ({e.reason})"
        ) from e

    try:
        # Parse JSON response from Ollama.
        obj = json.loads(data.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise OllamaError(f"Invalid JSON from Ollama: {e}") from e

    # Extract final assistant message content.
    content = (obj.get("message") or {}).get("content") or ""
    if not isinstance(content, str):
        raise OllamaError("Unexpected response shape from Ollama.")
    # Return trimmed text for CLI display.
    return content.strip()
