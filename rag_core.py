"""
Shared retrieval helpers for CLI and web app.
"""

from __future__ import annotations

from typing import Any, Dict, List

import chromadb

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "jazz_piano"
DEFAULT_K = 5


def get_collection(chroma_dir: str = CHROMA_DIR, name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(name)


def retrieve(collection, question: str, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
    """Return top-k semantic hits for a question."""
    results = collection.query(query_texts=[question], n_results=k)
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


def build_context_block(hits: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    lines = []
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        lines.append(f"[{i}] Article: {m['title']} | Section: {m['section']}")
        lines.append(hit["text"])
        lines.append("")
    return "\n".join(lines)


def hits_to_chunks(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for hit in hits:
        m = hit["metadata"]
        chunks.append(
            {
                "text": hit["text"],
                "title": m.get("title", ""),
                "section": m.get("section", ""),
                "source_url": m.get("source_url", ""),
                "similarity": round(1 - float(hit["distance"]), 4),
            }
        )
    return chunks


def hits_to_sources(hits: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sources = []
    seen = set()
    for hit in hits:
        m = hit["metadata"]
        key = (m.get("title", ""), m.get("section", ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": key[0],
                "section": key[1],
                "url": m.get("source_url", ""),
            }
        )
    return sources
