"""
Step 2: Embed chunks and store them in ChromaDB.

WHAT IS HAPPENING HERE?
  1. Each chunk of text is converted into a *vector* (a list of 384 numbers)
     by the sentence-transformers model "all-MiniLM-L6-v2".  This model was
     trained so that texts with similar *meaning* produce vectors that are
     close together in 384-dimensional space.

  2. ChromaDB stores each vector alongside the original text and its metadata.
     Under the hood it builds an HNSW index (Hierarchical Navigable Small World)
     so that nearest-neighbor lookups ("find the 5 vectors closest to this
     query vector") are fast even with thousands of chunks.

  3. The database is persisted to disk at ./chroma_db/ so you only need to
     run this script once.  The query script reads from the same directory.

WHY all-MiniLM-L6-v2?
  - It runs locally (no API key, no cost).
  - 384 dimensions is small and fast, but still captures meaning well.
  - It is ChromaDB's default embedding function, so zero extra config.
"""

import json
import chromadb

# --- Configuration ---

CHUNKS_FILE = "chunks.json"       # Input: output of build_chunks.py
CHROMA_DIR = "./chroma_db"        # Where the vector DB is persisted on disk
COLLECTION_NAME = "jazz_piano"    # Name of the collection inside ChromaDB


def main():
    """
    Load chunks from JSON, embed them, and store in ChromaDB.

    Steps:
      1. Read chunks.json (produced by build_chunks.py)
      2. Connect to (or create) a persistent ChromaDB instance on disk
      3. Delete any existing collection with the same name (for clean rebuilds)
      4. Create a new collection configured for cosine similarity
      5. Add all chunks in batches of 100 (ChromaDB auto-embeds each document
         using its default model: all-MiniLM-L6-v2 via ONNX runtime)
      6. Run a quick sanity-check query to verify the index works
    """
    # --- Load chunks ---
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # --- Connect to ChromaDB ---
    # PersistentClient saves the database to disk so it survives between runs.
    # The first time you run this, it creates the directory.  On subsequent
    # runs it opens the existing database.
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # --- Clear old data for a clean rebuild ---
    # If the collection already exists (from a previous run), delete it so we
    # start fresh.  This avoids duplicate or stale chunks.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")

    # --- Create collection ---
    # "hnsw:space": "cosine" tells ChromaDB to use cosine similarity when
    # comparing vectors.  Cosine measures the angle between vectors, ignoring
    # magnitude -- so it focuses purely on semantic direction.
    # Other options: "l2" (Euclidean distance) or "ip" (inner product).
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # --- Add chunks in batches ---
    # ChromaDB embeds each document automatically when you pass it as a string
    # in the 'documents' parameter.  We batch to avoid memory spikes.
    # Each chunk gets a unique ID, its text (for embedding + retrieval),
    # and its metadata dict (for filtering and display at query time).
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        documents = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  Added batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"\nDone! Collection '{COLLECTION_NAME}' has {collection.count()} vectors.")
    print(f"Persisted to {CHROMA_DIR}/")

    # --- Sanity check ---
    # Run a quick query to verify the index is working.  The query text is
    # embedded with the same model, and ChromaDB returns the closest vectors.
    # Lower distance = more similar (cosine distance, not cosine similarity).
    print("\n--- Sanity check: querying 'tritone substitution' ---")
    results = collection.query(query_texts=["tritone substitution"], n_results=3)
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  [{dist:.4f}] {meta['title']} > {meta['section']}")
        print(f"    {doc[:120]}...")

    # Build the 3D PCA map used by the portfolio visualization.
    try:
        from build_embeddings_3d import main as build_3d

        print("\n--- Building 3D embedding map ---")
        build_3d()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[warn] Could not build embeddings_3d.json: {exc}")


if __name__ == "__main__":
    main()
