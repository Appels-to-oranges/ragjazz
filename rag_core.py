"""
Shared retrieval helpers for CLI and web app.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "jazz_piano"
DEFAULT_K = 5
# Pull a wider candidate pool, then diversify down to k.
CANDIDATE_MULTIPLIER = 4
MAX_PER_ARTICLE = 2
MAX_PER_SECTION = 1


def clean_chunk_text(text: str) -> str:
    """Strip leading junk left by overlap splits (e.g. ' . The Coltrane...')."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^[\s\.\,\;\:\!\?\-\u2013\u2014\)\]\}]+", "", cleaned)
    return cleaned.strip()


def get_collection(chroma_dir: str = CHROMA_DIR, name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(name)


def _diversify_hits(
    hits: List[Dict[str, Any]],
    k: int,
    max_per_article: int = MAX_PER_ARTICLE,
    max_per_section: int = MAX_PER_SECTION,
) -> List[Dict[str, Any]]:
    """
    Reduce near-duplicate retrieval.

    Keeps ranking order, but caps how many chunks come from the same
    article / section, and skips adjacent chunk_index neighbors from the
    same article (overlap twins from the splitter).
    """
    selected: List[Dict[str, Any]] = []
    per_article: Dict[str, int] = {}
    per_section: Dict[Tuple[str, str], int] = {}
    taken_indexes: Set[Tuple[str, int]] = set()

    for hit in hits:
        if len(selected) >= k:
            break
        meta = hit.get("metadata") or {}
        title = str(meta.get("title", ""))
        section = str(meta.get("section", ""))
        try:
            chunk_index = int(meta.get("chunk_index", -1))
        except (TypeError, ValueError):
            chunk_index = -1

        if per_article.get(title, 0) >= max_per_article:
            continue
        if per_section.get((title, section), 0) >= max_per_section:
            continue
        if chunk_index >= 0 and (
            (title, chunk_index - 1) in taken_indexes
            or (title, chunk_index + 1) in taken_indexes
        ):
            continue

        selected.append(hit)
        per_article[title] = per_article.get(title, 0) + 1
        per_section[(title, section)] = per_section.get((title, section), 0) + 1
        if chunk_index >= 0:
            taken_indexes.add((title, chunk_index))

    # If diversity rules left us short, backfill with remaining top hits.
    if len(selected) < k:
        selected_ids = {id(h) for h in selected}
        for hit in hits:
            if len(selected) >= k:
                break
            if id(hit) in selected_ids:
                continue
            selected.append(hit)

    return selected


def retrieve(
    collection,
    question: str,
    k: int = DEFAULT_K,
    *,
    diversify: bool = True,
    candidate_multiplier: int = CANDIDATE_MULTIPLIER,
) -> List[Dict[str, Any]]:
    """
    Return top-k semantic hits for a question.

    When diversify=True (default), over-fetches candidates and applies
    article/section/adjacency caps so the context is less repetitive.
    """
    fetch_k = k * candidate_multiplier if diversify else k
    # Chroma errors if n_results > collection size; clamp defensively.
    try:
        fetch_k = min(fetch_k, max(collection.count(), 1))
    except Exception:
        pass

    results = collection.query(query_texts=[question], n_results=fetch_k)
    hits = []
    ids = results.get("ids") or [[]]
    for doc, meta, dist, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        ids[0],
    ):
        hits.append(
            {
                "id": chunk_id,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )

    if diversify:
        return _diversify_hits(hits, k=k)
    return hits[:k]


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
                "id": hit.get("id"),
                "text": clean_chunk_text(hit["text"]),
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
