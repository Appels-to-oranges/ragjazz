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
        "note": (
            "PCA overview of the corpus. Neighbors are chosen by cosine similarity "
            "in 384-d embedding space — they may not look nearby in this 3D projection."
        ),
        "query": query,
        "neighbors": neighbors,
    }
