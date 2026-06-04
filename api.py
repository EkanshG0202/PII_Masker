"""
Indian PII Masker — REST API
============================
Wraps indian_pii_masker.py in a FastAPI application.

Endpoints
─────────
  POST /mask          Mask PII in a text string
  POST /mask/batch    Mask PII in multiple texts at once
  GET  /health        Liveness check + resource snapshot

Setup
─────
    pip install fastapi uvicorn psutil

Run
───
    uvicorn api:app --reload --port 8000

    # or directly:
    python api.py
"""

from __future__ import annotations

import os
import time

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from indian_pii_masker import mask_pii, ResourceMonitor


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Indian PII Masker",
    description="Detects and masks Indian PII (Aadhaar, PAN, GST, IFSC, Phone, Email, and more).",
    version="1.0.0",
)

_proc = psutil.Process(os.getpid())


# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================

class MaskRequest(BaseModel):
    text: str = Field(..., description="Input text to mask.")
    monitor: bool = Field(False, description="Include resource usage stats in the response.")


class MaskResponse(BaseModel):
    masked_text: str
    resources: dict | None = None


class BatchMaskRequest(BaseModel):
    texts: list[str] = Field(..., description="List of input texts to mask.")
    monitor: bool = Field(False, description="Include per-item resource usage stats.")


class BatchMaskResponse(BaseModel):
    results: list[MaskResponse]


class HealthResponse(BaseModel):
    status: str
    uptime_s: float
    memory_rss_mb: float
    cpu_percent: float
    threads: int


# =========================================================
# STARTUP
# =========================================================

_start_time = time.perf_counter()


# =========================================================
# ENDPOINTS
# =========================================================

@app.post(
    "/mask",
    response_model=MaskResponse,
    summary="Mask PII in a single text",
)
def mask_single(req: MaskRequest) -> MaskResponse:
    """
    Accepts a text string and returns it with all detected Indian PII
    replaced by labelled placeholders (e.g. `[PAN]`, `[AADHAAR]`).
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="'text' must not be empty.")

    with ResourceMonitor("mask_pii", print_report=False) as mon:
        masked = mask_pii(req.text)

    return MaskResponse(
        masked_text=masked,
        resources=mon.stats if req.monitor else None,
    )


@app.post(
    "/mask/batch",
    response_model=BatchMaskResponse,
    summary="Mask PII in multiple texts",
)
def mask_batch(req: BatchMaskRequest) -> BatchMaskResponse:
    """
    Accepts a list of text strings and returns each one masked.
    Useful for processing records from a CSV or database in one call.
    """
    if not req.texts:
        raise HTTPException(status_code=422, detail="'texts' list must not be empty.")

    results = []
    for text in req.texts:
        with ResourceMonitor("mask_pii", print_report=False) as mon:
            masked = mask_pii(text)
        results.append(MaskResponse(
            masked_text=masked,
            resources=mon.stats if req.monitor else None,
        ))

    return BatchMaskResponse(results=results)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
def health() -> HealthResponse:
    """Returns server status and a current resource snapshot."""
    mem   = _proc.memory_info()
    cpu   = _proc.cpu_percent(interval=0.1)
    return HealthResponse(
        status="ok",
        uptime_s=round(time.perf_counter() - _start_time, 2),
        memory_rss_mb=round(mem.rss / 1024 / 1024, 2),
        cpu_percent=cpu,
        threads=_proc.num_threads(),
    )


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)