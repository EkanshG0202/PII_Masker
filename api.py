"""
Indian PII Masker — REST API
============================
Wraps the Two-Pass PII Masking Pipeline in a FastAPI application.
Pass 1: Strict Regex (indian_pii_masker.py)
Pass 2: Contextual NLP (indian_gliner_pii_masker.py)

Endpoints
─────────
  POST /mask          Mask PII in a text string
  POST /mask/batch    Mask PII in multiple texts at once
  GET  /health        Liveness check + resource snapshot

Setup
─────
    pip install fastapi uvicorn psutil gliner presidio-analyzer presidio-anonymizer

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
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# Import Pass 1 (Strict IDs) and the utility monitor
from indian_pii_masker import mask_pii, ResourceMonitor
# Import Pass 2 (Contextual NLP)
from indian_pii_text_masker import mask_text_entities


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Indian PII Masker",
    description="Detects and masks Indian PII using a Two-Pass Pipeline (Regex -> GLiNER).",
    version="2.0.0",
)

_proc = psutil.Process(os.getpid())


# =========================================================
# MIDDLEWARE
# =========================================================

@app.middleware("http")
async def log_response_time(request: Request, call_next):
    """
    Middleware to calculate and print the total response time for every request.
    Also injects the timing into the response headers.
    """
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    process_time = time.perf_counter() - start_time
    print(f"[{request.method}] {request.url.path} - Total Response Time: {process_time:.4f} seconds")
    
    # Optional: Attach the process time to the response headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


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
    summary="Mask PII in a single text using a two-pass pipeline",
)
def mask_single(req: MaskRequest) -> MaskResponse:
    """
    Executes a Two-Pass Redaction:
    1. Evaluates strict structured IDs via regex.
    2. Passes the redacted string to GLiNER to capture context-heavy names and addresses.
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="'text' must not be empty.")

    # Wrap the entire two-step process in one monitor block to get cumulative latency/CPU metrics
    with ResourceMonitor("two_pass_masking", print_report=False) as mon:
        step1_masked = mask_pii(req.text, monitor=False)
        final_masked = mask_text_entities(step1_masked, monitor=False)

    return MaskResponse(
        masked_text=final_masked,
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
    Routes each string through both the strict ID and NLP masking modules.
    """
    if not req.texts:
        raise HTTPException(status_code=422, detail="'texts' list must not be empty.")

    results = []
    for text in req.texts:
        with ResourceMonitor("two_pass_batch_item", print_report=False) as mon:
            step1_masked = mask_pii(text, monitor=False)
            final_masked = mask_text_entities(step1_masked, monitor=False)
            
        results.append(MaskResponse(
            masked_text=final_masked,
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