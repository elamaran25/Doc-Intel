"""
FastAPI serving layer. Run with:
    uvicorn serving.app:app --reload --port 8000
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from extraction.vlm_client import VLMConfig
from ingestion.loader import preprocess_document
from validation.agent import run_pipeline

load_dotenv()

app = FastAPI(title="Document Intelligence Pipeline", version="0.1.0")

DEFAULT_CONFIG = VLMConfig(
    provider=os.getenv("VLM_PROVIDER", "groq"),
    model_name=os.getenv("VLM_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct"),
    base_url=os.getenv("VLM_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.getenv("VLM_API_KEY", ""),
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        page_content = preprocess_document(tmp_path)
        result = await run_pipeline(page_content, DEFAULT_CONFIG)
        return JSONResponse(content=result.model_dump(mode="json"))
    finally:
        os.unlink(tmp_path)
