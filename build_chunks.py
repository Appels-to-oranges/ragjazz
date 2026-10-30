"""
Step 1: Chunk articles and attach metadata.

WHY CHUNK?
  Embedding models and LLMs work best on short, focused passages.
  A full article is too long to embed as a single vector -- the meaning gets
  diluted.  By splitting into overlapping ~500-char chunks we get vectors that
  capture specific ideas ("tritone substitution resolves to I") rather than
  vague article-level topics.

  Overlap (100 chars) ensures that sentences split across a boundary still
  appear intact in at least one chunk.

METADATA attached to every chunk:
  - title        : article it came from (for filtering & attribution)
  - source_url   : Wikipedia link (for citations)
  - section      : nearest section heading (for context)
  - chunk_index  : position in the article (for ordering)
"""

import json
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---

ARTICLES_DIR = "articles"
OUTPUT_FILE = "chunks.json"
# Slightly larger chunks + less overlap = fewer near-duplicate neighbors
# while still keeping ideas intact across boundaries.
CHUNK_SIZE = 650       # Max characters per chunk
CHUNK_OVERLAP = 50     # Characters shared between consecutive chunks

# Sections that contain no useful content for RAG (references, links, etc.)
SKIP_SECTIONS = {
    "See also", "References", "External links", "Sources",
    "Further reading", "Notes", "Bibliography",
}


def clean_chunk_text(text: str) -> str:
    """Strip leading junk left by overlap splits (e.g. ' . The Coltrane...')."""
    if not text:
        return ""
    cleaned = text.strip()
    # Drop orphan punctuation / connectors at the start of a chunk.
    cleaned = re.sub(r"^[\s\.\,\;\:\!\?\-\u2013\u2014\)\]\}]+", "", cleaned)
    cleaned = cleaned.lstrip()
    # If we stripped into lowercase mid-sentence debris, still fine to return.
    return cleaned.strip()

# The splitter tries each separator in order.  It picks the first one that
# produces pieces under CHUNK_SIZE.  So it prefers paragraph breaks, then
# line breaks, then sentence ends, then word boundaries, then character-level.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def build_section_map(full_text):
    """
    Build a list of (char_offset, section_name) markers from wiki-style headers.

    Some articles (downloaded via the direct MediaWiki API) store ALL their text
    in a single 'full_text' field with '== Section ==' style headers inline.
    This function scans for those headers and records where each section starts,
    so we can later determine which section any given character position falls in.

    Returns a list of tuples: [(0, "Introduction"), (450, "History"), (1200, "Theory"), ...]
    The list is in ascending offset order.
    """
    header_re = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)
    markers = [(0, "Introduction")]
    for m in header_re.finditer(full_text):
        markers.append((m.start(), m.group(2)))
    return markers


def section_at(markers, char_offset):
    """
    Given a list of (offset, section_name) markers, determine which section
    contains the given character position.

    Walks through markers in order.  The last marker whose offset is <= char_offset
    is the active section.  For example, if markers are [(0, "Intro"), (500, "History")]
    and char_offset is 600, the result is "History".
    """
    current = "Introduction"
    for offset, name in markers:
        if offset > char_offset:
            break
        current = name
    return current


def collect_section_texts(sections, depth=0):
    """
    Recursively walk the nested section tree and collect (title, text) pairs.

    Used for articles downloaded via the wikipedia-api library, where each section
    is a dict with 'title', 'text', and 'subsections' (a list of child sections).
    Skips boilerplate sections like "References" and "See also".

    Returns a flat list: [("Learning jazz piano", "Mastering the various..."), ...]
    """
    results = []
    for sec in sections:
        title = sec.get("title", "")
        text = sec.get("text", "")
        if title in SKIP_SECTIONS:
            continue
        if text.strip():
            results.append((title, text))
        for sub in sec.get("subsections", []):
            sub_title = sub.get("title", "")
            sub_text = sub.get("text", "")
            if sub_title in SKIP_SECTIONS:
                continue
            if sub_text.strip():
                results.append((sub_title, sub_text))
            results.extend(collect_section_texts(sub.get("subsections", []), depth + 1))
    return results


def has_section_text(sections):
    """
    Detect which article format we're dealing with.

    Returns True if the sections themselves carry text content (wikipedia-api format).
    Returns False if sections are just title/depth metadata and the real text lives
    in full_text (direct MediaWiki API format).
    """
    for sec in sections:
        if sec.get("text", "").strip():
            return True
        for sub in sec.get("subsections", []):
            if sub.get("text", "").strip():
                return True
    return False


def chunk_article(article):
    """
    Split a single article into chunks, each with metadata.

    Handles two different article formats:
      1. wikipedia-api format: sections have their own 'text' field.
         We chunk the intro paragraph, then chunk each section's text separately
         so that section attribution stays accurate.

      2. Direct MediaWiki API format: all text is in 'full_text' with inline
         '== Header ==' markers.  We strip the headers, chunk the cleaned text,
         then map each chunk back to its section by finding where the chunk's text
         appears in the original (using character offsets from build_section_map).

    Returns a list of chunk dicts: [{"text": "...", "metadata": {...}}, ...]
    """
    meta = article["metadata"]
    title = meta["title"]
    url = meta.get("url", meta.get("source_url", ""))
    sections = article.get("sections", [])
    full_text = article.get("full_text", "")

    chunks = []

    if has_section_text(sections):
        # --- Format 1: wikipedia-api (sections carry text) ---

        # Collect all (section_title, section_text) pairs from the nested tree
        intro_parts = collect_section_texts(sections)

        # The intro/summary is the text before the first section header.
        # We grab it from the start of full_text up to the first double-newline block.
        intro_text = full_text.split("\n\n")[0] if full_text else ""
        if intro_text.strip():
            for i, chunk_text in enumerate(splitter.split_text(intro_text)):
                cleaned = clean_chunk_text(chunk_text)
                if not cleaned:
                    continue
                chunks.append({
                    "text": cleaned,
                    "metadata": {
                        "title": title,
                        "source_url": url,
                        "section": "Introduction",
                        "chunk_index": len(chunks),
                    },
                })

        # Chunk each section independently so metadata stays accurate
        for sec_title, sec_text in intro_parts:
            for chunk_text in splitter.split_text(sec_text):
                cleaned = clean_chunk_text(chunk_text)
                if not cleaned:
                    continue
                chunks.append({
                    "text": cleaned,
                    "metadata": {
                        "title": title,
                        "source_url": url,
                        "section": sec_title,
                        "chunk_index": len(chunks),
                    },
                })
    else:
        # --- Format 2: Direct MediaWiki API (text with inline == headers ==) ---

        # Strip section headers from the text so they don't end up inside chunks
        clean = re.sub(r"^={2,}\s*.+?\s*={2,}\s*$", "", full_text, flags=re.MULTILINE)
        lines = clean.strip().split("\n")
        filtered_lines = [
            ln for ln in lines
            if ln.strip() and ln.strip() not in SKIP_SECTIONS
        ]
        clean_text = "\n".join(filtered_lines)

        # Build a map of where each section starts in the ORIGINAL text
        # (before we stripped headers), so we can look up sections by position
        section_markers = build_section_map(full_text)

        # Split the cleaned text into chunks
        raw_chunks = splitter.split_text(clean_text)

        # For each chunk, find its position in the original text to determine
        # which section it belongs to.  We search forward (search_start) to
        # avoid matching earlier occurrences of the same text.
        search_start = 0
        for chunk_text in raw_chunks:
            # Use first 80 chars of the chunk to locate it in the original
            pos = full_text.find(chunk_text[:80], search_start)
            if pos == -1:
                pos = search_start
            sec_name = section_at(section_markers, pos)
            if sec_name in SKIP_SECTIONS:
                continue
            search_start = max(search_start, pos + 1)

            cleaned = clean_chunk_text(chunk_text)
            if not cleaned:
                continue
            chunks.append({
                "text": cleaned,
                "metadata": {
                    "title": title,
                    "source_url": url,
                    "section": sec_name,
                    "chunk_index": len(chunks),
                },
            })

    return chunks


def main():
    """
    Load all article JSON files, chunk each one, and write the combined
    result to chunks.json.  Prints per-article stats so you can verify
    coverage.
    """
    all_chunks = []
    article_stats = {}

    # Process each article file (skip the _manifest.json)
    for filename in sorted(os.listdir(ARTICLES_DIR)):
        if filename.startswith("_") or not filename.endswith(".json"):
            continue
        filepath = os.path.join(ARTICLES_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            article = json.load(f)

        chunks = chunk_article(article)
        article_stats[article["metadata"]["title"]] = len(chunks)
        all_chunks.extend(chunks)

    # Write all chunks to a single JSON file for inspection and for
    # build_vectorstore.py to consume
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total chunks: {len(all_chunks)}")
    print(f"\nPer article:")
    for title, count in sorted(article_stats.items()):
        print(f"  {title}: {count} chunks")

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
