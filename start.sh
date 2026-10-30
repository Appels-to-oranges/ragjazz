#!/bin/sh
set -e
if [ ! -d "./chroma_db" ]; then
  echo "Building Chroma vector store..."
  python build_vectorstore.py
fi
if [ ! -f "./embeddings_3d.json" ]; then
  echo "Building 3D embedding map..."
  python build_embeddings_3d.py
fi
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
