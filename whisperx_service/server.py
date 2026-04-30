# WhisperX transcription microservice.
#
# Exposes a small HTTP API consumed by the main web app:
#   GET  /healthz       liveness + model/device info
#   POST /transcribe    multipart audio upload -> {text, language, segments}
#
# The model is loaded once on startup so subsequent requests are warm.
from __future__ import annotations

import os
import tempfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import whisperx

try:
    import torch  # type: ignore

    _CUDA_AVAILABLE = bool(getattr(torch.cuda, "is_available", lambda: False)())
except Exception:
    _CUDA_AVAILABLE = False


def _device() -> str:
    requested = (os.getenv("WHISPERX_DEVICE", "") or "").strip().lower()
    if requested in {"cuda", "cpu"}:
        return requested
    return "cuda" if _CUDA_AVAILABLE else "cpu"


def _compute_type(device: str) -> str:
    requested = (os.getenv("WHISPERX_COMPUTE_TYPE", "") or "").strip()
    if requested:
        return requested
    return "float16" if device == "cuda" else "int8"


MODEL_NAME = (os.getenv("WHISPERX_MODEL", "tiny") or "tiny").strip() or "tiny"
DEVICE = _device()
COMPUTE_TYPE = _compute_type(DEVICE)
BATCH_SIZE = max(1, int(os.getenv("WHISPERX_BATCH_SIZE", "8") or "8"))


app = FastAPI(title="WhisperX Transcription")
_model: Any = None


def _get_model() -> Any:
    global _model
    if _model is None:
        _model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
    return _model


@app.on_event("startup")
def _warmup() -> None:
    # Pre-load weights so the first user request doesn't pay the cold-start cost.
    try:
        _get_model()
    except Exception as exc:
        # Fail soft: surface the error on the next /transcribe call instead of
        # crashing the process on a transient model-download issue.
        print(f"[whisperx] warmup failed: {exc}", flush=True)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "loaded": _model is not None,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty audio upload.")

    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        audio = whisperx.load_audio(tmp_path)
        kwargs: dict[str, Any] = {"batch_size": BATCH_SIZE}
        lang = (language or "").strip().lower()
        if lang and lang != "auto":
            kwargs["language"] = lang
        result = _get_model().transcribe(audio, **kwargs)
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    segments = result.get("segments") or []
    text = " ".join(str(seg.get("text", "") or "").strip() for seg in segments).strip()
    return {
        "text": text,
        "language": str(result.get("language", "") or ""),
        "segments": segments,
    }
