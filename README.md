# RAG Jazz

A jazz piano / music theory Q&A app grounded in a curated Wikipedia knowledge base via retrieval-augmented generation.

Ask a question, get matching source passages, and (when configured) a natural-language answer from Groq's free tier or local Ollama. If the LLM is unavailable, the app falls back to retrieval-only.

## Knowledge base

**39 articles** (~814 chunks), focused on jazz piano theory:

### Foundations
Interval (music), Triad (music), Key signature, Major scale, Minor scale

### Core piano & styles
Jazz piano, Stride piano, Boogie-woogie, Comping (jazz), Block chord, Walking bass

### Harmony & progressions
Jazz harmony, Chord (music), Seventh chord, Extended chord, Slash chord, Chord progression, ii–V–I progression, Secondary dominant, Tritone substitution, Coltrane changes, Backdoor progression, Turnaround (music), Rhythm changes, Twelve-bar blues, Circle of fifths

### Scales & improvisation
Jazz scale, Bebop scale, Blues scale, Pentatonic scale, Altered scale, Modal jazz, Jazz improvisation, Chord-scale system

### Voicing & voice leading
Voicing (music), Voice leading, Upper structure, Quartal harmony, Altered chord

Retrieval over-fetches candidates and diversifies by article/section so answers are less repetitive. Chunking uses 650-char chunks with 50-char overlap.

### Embedding-space visualization
The UI includes a Three.js PCA map of the knowledge base. Neighbors are chosen by cosine similarity in 384-d and highlighted on the map (3D distance is only an approximation). Rebuild with:

```bash
python build_embeddings_3d.py
```

(`build_vectorstore.py` also builds it automatically.)

## Local setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python build_vectorstore.py
```

`chunks.json` is already committed. Rebuild after changing articles:

```bash
python download_articles.py   # or download only new titles
python rebuild_manifest.py
python build_chunks.py
python build_vectorstore.py
```

### Web app

```bash
copy .env.example .env   # optional: set GROQ_API_KEY or LLM_PROVIDER=ollama
uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000  
Plain-language explainer: http://127.0.0.1:8000/how-it-works

### CLI

```bash
python query.py "What is a tritone substitution?"
python query.py --no-llm "circle of fifths"
python query.py --model mistral "stride piano"
```

Local LLM answers require [Ollama](https://ollama.com/download):

```bash
ollama pull llama3.2
```

## Environment

See `.env.example`.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `groq`, `ollama`, or `none` |
| `GROQ_API_KEY` | Groq free-tier key |
| `GROQ_MODEL` | Default `openai/gpt-oss-20b` |
| `OLLAMA_MODEL` | Default `llama3.2` |
| `RATE_LIMIT_RPM` | Per-IP anti-abuse limit (default 20) |

If Groq rate-limits or errors, the API returns retrieval results with a notice.

## Deploy

### Docker

```bash
docker build -t rag-jazz .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key -e LLM_PROVIDER=groq rag-jazz
```

The image builds the Chroma index at build time.

### Railway / Procfile

`start.sh` builds `chroma_db` on first boot if missing, then starts uvicorn. Set:

- `GROQ_API_KEY`
- `LLM_PROVIDER=groq` (or leave unset to auto-detect from the key)
- `PORT` (provided by the host)

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | FastAPI web server |
| `rag_core.py` | Shared retrieval helpers |
| `llm_provider.py` | Groq / Ollama / none |
| `static/` | Web UI |
| `query.py` | CLI query tool |
| `build_chunks.py` / `build_vectorstore.py` | Index pipeline |
| `articles/` | Source Wikipedia articles |

## Other scripts

- `download_articles.py` — Primary Wikipedia downloader (curated title list)
- `download_remaining.py` — MediaWiki REST fallback
- `rebuild_manifest.py` — Regenerates `articles/_manifest.json`
