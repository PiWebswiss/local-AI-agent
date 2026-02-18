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
    pass


_MODEL_SIZE_B_RE = re.compile(r"(?<!\d)(?P<size>[0-9]+)b(?!\w)", re.IGNORECASE)


def validate_model(model: str, *, max_b: int = 4) -> str:
    model = (model or "").strip()
    if not model:
        raise OllamaError("Model name is empty.")

    if max_b > 0:
        match = _MODEL_SIZE_B_RE.search(model)
        if match:
            size_b = int(match.group("size"))
            if size_b > max_b:
                raise OllamaError(f"Model '{model}' exceeds max size ({max_b}b).")
    return model


def validate_gemma3_model(model: str, *, max_b: int = 4) -> str:
    return validate_model(model, max_b=max_b)


def _normalize_host(host: str) -> str:
    host = (host or "").strip()
    if not host:
        return "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


@dataclass(frozen=True)
class OllamaConfig:
    host: str
    model: str
    timeout_s: float = 120.0

    @staticmethod
    def from_env() -> "OllamaConfig":
        host = _normalize_host(os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
        model = validate_model(model, max_b=int(os.getenv("OLLAMA_MAX_B", "4")))
        timeout_s = float(os.getenv("OLLAMA_TIMEOUT_S", "120"))
        return OllamaConfig(host=host, model=model, timeout_s=timeout_s)


def chat(
    *,
    config: OllamaConfig,
    messages: list[dict[str, str]],
    options: dict[str, Any] | None = None,
) -> str:
    url = f"{config.host}/api/chat"
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options

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
        with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise OllamaError(f"Ollama HTTP {e.code}: {body or e.reason}") from e
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Failed to reach Ollama at {config.host}. "
            f"Is Ollama running and accessible? ({e.reason})"
        ) from e

    try:
        obj = json.loads(data.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise OllamaError(f"Invalid JSON from Ollama: {e}") from e

    content = (obj.get("message") or {}).get("content") or ""
    if not isinstance(content, str):
        raise OllamaError("Unexpected response shape from Ollama.")
    return content.strip()
