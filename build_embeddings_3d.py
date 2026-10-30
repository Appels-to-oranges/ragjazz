"""
Project chunk embeddings into 3D with PCA for the portfolio visualization.

Reads vectors from Chroma, fits a 3-component PCA (numpy SVD), and writes:
  - embeddings_3d.json  — point cloud + PCA params for projecting queries
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import numpy as np

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "jazz_piano"
OUTPUT_FILE = "embeddings_3d.json"
N_COMPONENTS = 3


def pca_fit_transform(matrix: np.ndarray, n_components: int = 3):
    """Center + SVD PCA. Returns coords, mean, components (n_components x dim)."""
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    coords = centered @ components.T
    return coords, mean, components


def project(vectors: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (vectors - mean) @ components.T


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()
    if count == 0:
        raise SystemExit("Chroma collection is empty. Run build_vectorstore.py first.")

    print(f"Loading {count} embeddings from Chroma…")
    raw = collection.get(include=["embeddings", "documents", "metadatas"])
    ids = raw["ids"]
    embeddings = np.asarray(raw["embeddings"], dtype=np.float64)
    metadatas = raw["metadatas"]
    documents = raw["documents"]

    print(f"Fitting PCA ({embeddings.shape[1]}-d → {N_COMPONENTS}-d)…")
    coords, mean, components = pca_fit_transform(embeddings, N_COMPONENTS)

    # Scale for a nicer Three.js framing (roughly unit-ish spread).
    scale = float(np.percentile(np.linalg.norm(coords, axis=1), 95) or 1.0)
    coords_scaled = (coords / scale).tolist()

    titles = sorted({(m or {}).get("title", "Unknown") for m in metadatas})
    title_to_color_index = {title: i for i, title in enumerate(titles)}

    points = []
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] or {}
        title = meta.get("title", "Unknown")
        points.append(
            {
                "id": chunk_id,
                "x": coords_scaled[i][0],
                "y": coords_scaled[i][1],
                "z": coords_scaled[i][2],
                "title": title,
                "section": meta.get("section", ""),
                "color_index": title_to_color_index[title],
                "preview": (documents[i] or "")[:140],
            }
        )

    payload = {
        "method": "pca",
        "note": (
            "3D positions are a PCA projection of 384-d MiniLM embeddings for viewing. "
            "Nearest neighbors are still chosen by cosine similarity in the original space."
        ),
        "count": len(points),
        "titles": titles,
        "scale": scale,
        "pca": {
            "mean": mean.tolist(),
            "components": components.tolist(),
        },
        "points": points,
    }

    out = Path(OUTPUT_FILE)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {out} ({len(points)} points, {len(titles)} articles)")


if __name__ == "__main__":
    main()
