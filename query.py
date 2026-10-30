"""
Step 3: Query the RAG system (CLI).

USAGE:
  python query.py "What is a tritone substitution?"
  python query.py "How does stride piano differ from boogie-woogie?"
  python query.py --no-llm "circle of fifths"    # retrieval only, no LLM
"""

import argparse
import io
import os
import sys
import textwrap

from llm_provider import DEFAULT_OLLAMA_MODEL, generate_answer
from rag_core import (
    DEFAULT_K,
    build_context_block,
    get_collection,
    retrieve,
)

# Force UTF-8 output on Windows so music symbols (sharps, flats) render
# instead of crashing with a codec error.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Query the Jazz Piano RAG system")
    parser.add_argument("question", help="Your question about jazz piano or music theory")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve")
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument("--no-llm", action="store_true", help="Show retrieved chunks only, skip LLM")
    args = parser.parse_args()

    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    os.environ.setdefault("LLM_PROVIDER", "ollama")

    collection = get_collection()
    print(f"\nSearching {collection.count()} chunks for: \"{args.question}\"\n")

    hits = retrieve(collection, args.question, k=args.k)

    print("=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        sim = 1 - hit["distance"]
        print(f"\n[{i}] {m['title']} > {m['section']}  (similarity: {sim:.2%})")
        print("-" * 60)
        print(textwrap.fill(hit["text"], width=80))

    if args.no_llm:
        return

    print("\n" + "=" * 60)
    print("GENERATING ANSWER...")
    print("=" * 60)

    context = build_context_block(hits)
    answer, error = generate_answer(args.question, context, provider="ollama")

    if error:
        model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        print(f"\n[Ollama unavailable] {error}")
        print("\nYou can still see the retrieved chunks above.")
        print("To enable LLM answers, install Ollama and pull a model:")
        print("  1. Download from https://ollama.com/download")
        print(f"  2. ollama pull {model}")
        print("  3. Re-run this script")
        return

    print(f"\n{answer}")
    print("\n" + "-" * 60)
    print("Sources used:")
    seen = set()
    for hit in hits:
        m = hit["metadata"]
        key = (m["title"], m["section"])
        if key not in seen:
            seen.add(key)
            print(f"  - {m['title']} > {m['section']}  ({m.get('source_url', '')})")


if __name__ == "__main__":
    main()
