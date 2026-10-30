"""
Visualization helpers for embedding space.

Why neighbors looked far away before:
  Global PCA preserves variance axes, not local cosine neighborhoods.
  A point can be a top cosine neighbor in 384-d and still land far from
  the query in a 3-d PCA plot.

Fix used at query time:
  Build a *local* 3-d layout with classical MDS on cosine distances among
  the query, its retrieved neighbors, and a sample of other chunks.
  In that view, nearer ≈ higher cosine similarity by construction.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

EMBEDDINGS_3D_PATH = Path("embeddings_3d.json")
CONTEXT_SAMPLE = 50
RNG = random.Random(42)


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


def _l2_normalize(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    normalized = _l2_normalize(vectors)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return 1.0 - similarity


def classical_mds(distances: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Classical MDS (metric) from a symmetric distance matrix."""
    n = distances.shape[0]
    d2 = distances ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order][:n_components], 0.0)
    eigvecs = eigvecs[:, order][:, :n_components]
    coords = eigvecs * np.sqrt(eigvals)
    # Center query (index 0) at origin for a stable camera target.
    coords = coords - coords[0]
    spread = np.percentile(np.linalg.norm(coords, axis=1), 95) or 1.0
    return coords / spread


def build_local_cosine_visualization(
    question: str,
    hits: List[Dict[str, Any]],
    collection,
    context_sample: int = CONTEXT_SAMPLE,
) -> Dict[str, Any]:
    """
    Local cosine-MDS scene for one query.

    Returns query + neighbor + context coordinates where spatial nearness
    tracks cosine distance for this neighborhood.
    """
    meta_cloud = load_embeddings_3d()
    title_color = {}
    if meta_cloud:
        for i, title in enumerate(meta_cloud.get("titles") or []):
            title_color[title] = i

    query_vec = embed_texts([question])[0]

    neighbor_ids = [h.get("id") for h in hits if h.get("id")]
    neighbor_meta = {h.get("id"): h for h in hits if h.get("id")}

    # Sample additional corpus points for contrast (not the selected neighbors).
    all_ids = []
    if meta_cloud:
        all_ids = [p["id"] for p in meta_cloud.get("points", [])]
    else:
        all_ids = collection.get(include=[]).get("ids") or []

    neighbor_set = set(neighbor_ids)
    candidates = [i for i in all_ids if i not in neighbor_set]
    RNG.shuffle(candidates)
    context_ids = candidates[:context_sample]

    ordered_ids = neighbor_ids + context_ids
    if not ordered_ids:
        return {"available": False, "error": "No points available for visualization"}

    fetched = collection.get(ids=ordered_ids, include=["embeddings", "metadatas", "documents"])
    id_to_index = {chunk_id: i for i, chunk_id in enumerate(fetched["ids"])}
    # Keep a stable order matching ordered_ids (drop any missing).
    vectors = [query_vec]
    rows: List[Dict[str, Any]] = [
        {"role": "query", "id": "query", "title": "Your query", "section": "", "preview": question}
    ]

    for chunk_id in ordered_ids:
        if chunk_id not in id_to_index:
            continue
        i = id_to_index[chunk_id]
        vectors.append(np.asarray(fetched["embeddings"][i], dtype=np.float64))
        meta = fetched["metadatas"][i] or {}
        role = "neighbor" if chunk_id in neighbor_set else "context"
        hit = neighbor_meta.get(chunk_id)
        similarity = None
        if hit is not None:
            similarity = round(1 - float(hit.get("distance", 1)), 4)
        rows.append(
            {
                "role": role,
                "id": chunk_id,
                "title": meta.get("title", ""),
                "section": meta.get("section", ""),
                "preview": (fetched["documents"][i] or "")[:140],
                "similarity": similarity,
                "color_index": title_color.get(meta.get("title", ""), 0),
            }
        )

    matrix = np.vstack(vectors)
    distances = _cosine_distance_matrix(matrix)
    coords = classical_mds(distances, n_components=3)

    query_point = {
        "x": float(coords[0, 0]),
        "y": float(coords[0, 1]),
        "z": float(coords[0, 2]),
    }

    neighbors = []
    context = []
    for row, xyz in zip(rows[1:], coords[1:]):
        point = {
            **row,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        }
        if row["role"] == "neighbor":
            neighbors.append(point)
        else:
            context.append(point)

    # Preserve retrieval order for neighbors.
    by_id = {n["id"]: n for n in neighbors}
    neighbors_ordered = [by_id[i] for i in neighbor_ids if i in by_id]

    return {
        "available": True,
        "method": "cosine_mds",
        "metric": "cosine",
        "note": (
            "Local view arranged with multidimensional scaling on cosine distances "
            "for this query. Closer points ≈ higher cosine similarity. "
            "(A global PCA map cannot reliably keep cosine neighbors nearby.)"
        ),
        "query": query_point,
        "neighbors": neighbors_ordered,
        "context": context,
    }
