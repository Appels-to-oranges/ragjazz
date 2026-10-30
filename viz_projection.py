"""
Visualization helpers for embedding space.

Two views are returned for each query:
  - pca: stable global PCA coordinates (corpus overview; neighbors may look far)
  - cosine: local MDS on cosine distances (neighbors sit near the query)
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

EMBEDDINGS_3D_PATH = Path("embeddings_3d.json")
CONTEXT_SAMPLE = 100
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
    coords = coords - coords[0]
    spread = np.percentile(np.linalg.norm(coords, axis=1), 95) or 1.0
    return coords / spread


def _project_query_pca(query_vec: np.ndarray, meta_cloud: Dict[str, Any]) -> Dict[str, float]:
    mean = np.asarray(meta_cloud["pca"]["mean"], dtype=np.float64)
    components = np.asarray(meta_cloud["pca"]["components"], dtype=np.float64)
    scale = float(meta_cloud.get("scale") or 1.0)
    coords = (query_vec - mean) @ components.T
    coords = coords / scale
    return {"x": float(coords[0]), "y": float(coords[1]), "z": float(coords[2])}


def _neighbor_base(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    neighbors = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        neighbors.append(
            {
                "id": hit.get("id"),
                "title": meta.get("title", ""),
                "section": meta.get("section", ""),
                "similarity": round(1 - float(hit.get("distance", 1)), 4),
            }
        )
    return neighbors


def build_pca_view(
    query_vec: np.ndarray,
    hits: List[Dict[str, Any]],
    meta_cloud: Dict[str, Any],
) -> Dict[str, Any]:
    by_id = {p["id"]: p for p in meta_cloud.get("points", [])}
    title_color = {
        title: i for i, title in enumerate(meta_cloud.get("titles") or [])
    }
    query = _project_query_pca(query_vec, meta_cloud)
    neighbors = []
    for base in _neighbor_base(hits):
        point = by_id.get(base["id"]) if base.get("id") else None
        entry = dict(base)
        entry["color_index"] = title_color.get(base.get("title", ""), 0)
        if point:
            entry.update(
                {
                    "x": point["x"],
                    "y": point["y"],
                    "z": point["z"],
                    "color_index": point.get("color_index", entry["color_index"]),
                }
            )
        neighbors.append(entry)
    return {
        "method": "pca",
        "note": (
            "Stable PCA overview of the full corpus. Neighbors are still chosen by "
            "cosine similarity in 384-d — they may not look nearby in this projection."
        ),
        "query": query,
        "neighbors": neighbors,
        "context": [],
    }


def build_cosine_view(
    question: str,
    query_vec: np.ndarray,
    hits: List[Dict[str, Any]],
    collection,
    meta_cloud: Optional[Dict[str, Any]],
    context_sample: int = CONTEXT_SAMPLE,
) -> Dict[str, Any]:
    title_color = {}
    if meta_cloud:
        for i, title in enumerate(meta_cloud.get("titles") or []):
            title_color[title] = i

    neighbor_ids = [h.get("id") for h in hits if h.get("id")]
    neighbor_meta = {h.get("id"): h for h in hits if h.get("id")}

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
        return {
            "method": "cosine_mds",
            "note": "No points available for local cosine layout.",
            "query": {"x": 0.0, "y": 0.0, "z": 0.0},
            "neighbors": [],
            "context": [],
        }

    fetched = collection.get(ids=ordered_ids, include=["embeddings", "metadatas", "documents"])
    id_to_index = {chunk_id: i for i, chunk_id in enumerate(fetched["ids"])}

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

    by_id = {n["id"]: n for n in neighbors}
    neighbors_ordered = [by_id[i] for i in neighbor_ids if i in by_id]

    return {
        "method": "cosine_mds",
        "note": (
            "Local cosine layout: points arranged so closer ≈ higher cosine similarity "
            "for this query."
        ),
        "query": query_point,
        "neighbors": neighbors_ordered,
        "context": context,
    }


def build_query_visualization(
    question: str,
    hits: List[Dict[str, Any]],
    collection,
) -> Dict[str, Any]:
    meta_cloud = load_embeddings_3d()
    query_vec = embed_texts([question])[0]

    cosine_view = build_cosine_view(question, query_vec, hits, collection, meta_cloud)
    pca_view = None
    if meta_cloud and meta_cloud.get("pca"):
        pca_view = build_pca_view(query_vec, hits, meta_cloud)

    if not pca_view and not cosine_view.get("neighbors"):
        return {"available": False, "error": "Visualization unavailable"}

    return {
        "available": True,
        "metric": "cosine",
        "default_view": "pca",
        "views": {
            "pca": pca_view,
            "cosine": cosine_view,
        },
    }
