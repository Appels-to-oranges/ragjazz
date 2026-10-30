#!/sh
set -e
if [ ! -d "./chroma_db" ]; then
  echo "Building Chroma vector store..."
  python build_vectorstore.py
fi
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
