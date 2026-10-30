"""
RAG Jazz web app — FastAPI + retrieval with optional Groq/Ollama answers.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from llm_provider import generate_answer, resolve_provider
from rag_core import (
    DEFAULT_K,
    build_context_block,
    get_collection,
    hits_to_chunks,
    hits_to_sources,
    retrieve,
)
from viz_projection import (
    attach_neighbor_positions,
    load_embeddings_3d,
    project_query,
)

STATIC_DIR = Path(__file__).parent / "static"
RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_RPM", "20"))

app = FastAPI(title="RAG Jazz", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_collection = None
_rate_lock = Lock()
_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    k: int = Field(DEFAULT_K, ge=1, le=10)


class AskResponse(BaseModel):
    mode: str
    question: str
    answer: Optional[str] = None
    chunks: list
    sources: list
    notice: Optional[str] = None
    visualization: Optional[dict] = None


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(ip: str) -> None:
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[ip]
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SEC:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a moment and try again.",
            )
        bucket.append(now)


def _get_collection():
    global _collection
    if _collection is None:
        try:
            _collection = get_collection()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=(
                    "Vector store not ready. Run: python build_vectorstore.py "
                    f"({exc})"
                ),
            ) from exc
    return _collection


def _build_visualization(question: str, hits: list) -> Optional[dict]:
    try:
        query_point = project_query(question)
        neighbors, note = attach_neighbor_positions(hits)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "error": str(exc),
        }
    if not query_point:
        return {
            "available": False,
            "error": note or "3D map unavailable",
        }
    return {
        "available": True,
        "method": "pca",
        "metric": "cosine",
        "note": note,
        "query": query_point,
        "neighbors": neighbors,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    try:
        collection = _get_collection()
        count = collection.count()
        ready = True
    except HTTPException:
        count = 0
        ready = False
    viz = load_embeddings_3d()
    return {
        "ok": ready,
        "chunks": count,
        "provider": resolve_provider(),
        "viz_points": (viz or {}).get("count", 0),
    }


@app.get("/api/embeddings-3d")
def embeddings_3d():
    data = load_embeddings_3d()
    if not data:
        raise HTTPException(
            status_code=404,
            detail="embeddings_3d.json missing. Run: python build_embeddings_3d.py",
        )
    # Omit bulky PCA matrices from the browser payload.
    return {
        "method": data.get("method"),
        "note": data.get("note"),
        "count": data.get("count"),
        "titles": data.get("titles", []),
        "points": data.get("points", []),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    _enforce_rate_limit(_client_ip(request))

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    collection = _get_collection()
    hits = retrieve(collection, question, k=payload.k)
    chunks = hits_to_chunks(hits)
    sources = hits_to_sources(hits)
    context = build_context_block(hits)
    visualization = _build_visualization(question, hits)

    answer, error = generate_answer(question, context)
    if answer:
        return AskResponse(
            mode="llm",
            question=question,
            answer=answer,
            chunks=chunks,
            sources=sources,
            visualization=visualization,
        )

    notice = None
    if error and "none" not in error.lower():
        notice = "AI answer unavailable — showing matching passages instead."
    elif resolve_provider() == "none":
        notice = "Running in retrieval-only mode."

    return AskResponse(
        mode="retrieval",
        question=question,
        answer=None,
        chunks=chunks,
        sources=sources,
        notice=notice,
        visualization=visualization,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
