#!/usr/bin/env python3
"""
reranker.py — CrossEncoder HTTP service with FastAPI
"""

import os
import logging
from typing import List, Optional, Any, Dict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from sentence_transformers import CrossEncoder

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("reranker")

MODEL_NAME = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
HOST = os.environ.get("RERANKER_HOST", "0.0.0.0")  # Bind to all interfaces by default for container friendliness
PORT = int(os.environ.get("RERANKER_PORT", "8765"))

# --- Global State ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    logger.info(f"Loading CrossEncoder model: {MODEL_NAME}...")
    try:
        ml_models["encoder"] = CrossEncoder(MODEL_NAME)
        logger.info(f"✅ CrossEncoder '{MODEL_NAME}' ready.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # We don't raise here to allow app to start, but /rerank will fail gracefully
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(title="RAG Hybrid Reranker", version="2.0.0", lifespan=lifespan)

# --- Pydantic Models ---

class Chunk(BaseModel):
    id: str
    text: str
    filename: str
    # Optional fields that might be present in the input
    rrf_raw: Optional[float] = 0.0
    score: Optional[float] = 0.0
    rerank_score: Optional[float] = 0.0
    
    # Allow extra fields to be passed through (e.g. metadata)
    model_config = ConfigDict(extra='allow')

class RerankRequest(BaseModel):
    query: str
    chunks: List[Chunk]

# --- Endpoints ---

@app.get("/health")
def health_check():
    if "encoder" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/")
def rerank(req: RerankRequest):
    encoder = ml_models.get("encoder")
    if not encoder:
        raise HTTPException(status_code=503, detail="Model not loaded or initialization failed")

    if not req.chunks:
        return []

    try:
        # Prepare pairs for CrossEncoder
        pairs = [[req.query, c.text] for c in req.chunks]
        
        # Predict scores
        # scores is a numpy array or list of floats
        scores = encoder.predict(pairs).tolist()
        
        # Update chunks with scores
        for i, chunk in enumerate(req.chunks):
            chunk.rerank_score = float(scores[i])

        # Sort by rerank_score descending
        req.chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return req.chunks

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"🚀 Starting Reranker API on {HOST}:{PORT}")
    uvicorn.run("reranker:app", host=HOST, port=PORT, log_level="info", reload=False)
