"""
Project a query embedding into the precomputed PCA 3D space.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def attach_neighbor_positions(
    hits: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Enrich hit dicts with 3D coords from embeddings_3d.json when ids match.
    Returns (neighbors_for_viz, note).
    """
    data = load_embeddings_3d()
    if not data:
        return [], "3D map not built yet. Run: python build_embeddings_3d.py"

    by_id = {p["id"]: p for p in data.get("points", [])}
    neighbors = []
    for hit in hits:
        chunk_id = hit.get("id")
        point = by_id.get(chunk_id) if chunk_id else None
        meta = hit.get("metadata") or {}
        similarity = 1 - float(hit.get("distance", 1))
        entry = {
            "id": chunk_id,
            "title": meta.get("title", ""),
            "section": meta.get("section", ""),
            "similarity": round(similarity, 4),
        }
        if point:
            entry.update({"x": point["x"], "y": point["y"], "z": point["z"]})
        neighbors.append(entry)
    return neighbors, data.get("note")
