"""
Project a query into the precomputed PCA 3D space and attach neighbor coords.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

EMBEDDINGS_3D_PATH = Path("embeddings_3d.json")


@lru_cache(maxsize=1)
def load_embeddings_3d() -> Optional[Dict[str, Any]]:
    if not EMBEDDINGS_3D_PATH.exists():
        return None
    return json.loads(EMBEDDINGS_3D_PATH.read_text(encoding="utf-8"))


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed with Chroma's default MiniLM ONNX function (same as the index)."""
    from chromadb.utils import embedding_functions

    ef = embedding_functions.DefaultEmbeddingFunction()
    vectors = ef(texts)
    return np.asarray(vectors, dtype=np.float64)


def project_query(question: str) -> Optional[Dict[str, float]]:
    data = load_embeddings_3d()
    if not data or "pca" not in data:
        return None

    mean = np.asarray(data["pca"]["mean"], dtype=np.float64)
    components = np.asarray(data["pca"]["components"], dtype=np.float64)
    scale = float(data.get("scale") or 1.0)

    vector = embed_texts([question])[0]
    coords = (vector - mean) @ components.T
    coords = coords / scale
    return {"x": float(coords[0]), "y": float(coords[1]), "z": float(coords[2])}


def build_query_visualization(
    question: str,
    hits: List[Dict[str, Any]],
    collection=None,  # unused; kept for call-site compatibility
) -> Dict[str, Any]:
    data = load_embeddings_3d()
    if not data:
        return {
            "available": False,
            "error": "3D map not built yet. Run: python build_embeddings_3d.py",
        }

    query = project_query(question)
    if not query:
        return {"available": False, "error": "Could not project query into PCA space"}

    by_id = {p["id"]: p for p in data.get("points", [])}
    neighbors = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        chunk_id = hit.get("id")
        point = by_id.get(chunk_id) if chunk_id else None
        entry = {
            "id": chunk_id,
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "similarity": round(1 - float(hit.get("distance", 1)), 4),
        }
        if point:
            entry.update({"x": point["x"], "y": point["y"], "z": point["z"]})
        neighbors.append(entry)

    return {
        "available": True,
        "method": "pca",
        "metric": "cosine",
        "question": question,
        "note": (
            "PCA overview of the corpus. Neighbors are chosen by cosine similarity "
            "in 384-d embedding space — they may not look nearby in this 3D projection. "
            "Click a point for article, section, and cosine score."
        ),
        "query": query,
        "neighbors": neighbors,
    }


def chunk_info(chunk_id: str, question: Optional[str] = None) -> Dict[str, Any]:
    """Return chunk metadata and optional cosine similarity to a question."""
    data = load_embeddings_3d()
    point = None
    if data:
        point = next((p for p in data.get("points", []) if p.get("id") == chunk_id), None)

    from rag_core import get_collection

    collection = get_collection()
    fetched = collection.get(ids=[chunk_id], include=["documents", "metadatas", "embeddings"])
    if not fetched.get("ids"):
        raise KeyError(chunk_id)

    meta = (fetched["metadatas"][0] or {}) if fetched.get("metadatas") else {}
    doc = (fetched["documents"][0] or "") if fetched.get("documents") else ""
    info = {
        "id": chunk_id,
        "title": meta.get("title") or (point or {}).get("title") or "",
        "section": meta.get("section") or (point or {}).get("section") or "",
        "preview": (point or {}).get("preview") or doc[:180],
        "similarity": None,
    }

    q = (question or "").strip()
    embeddings = fetched.get("embeddings")
    if q and embeddings is not None and len(embeddings) > 0:
        chunk_vec = np.asarray(embeddings[0], dtype=np.float64)
        query_vec = embed_texts([q])[0]
        denom = float(np.linalg.norm(chunk_vec) * np.linalg.norm(query_vec))
        if denom > 0:
            info["similarity"] = round(float(np.dot(chunk_vec, query_vec) / denom), 4)

    return info

