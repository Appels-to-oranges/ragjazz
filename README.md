# RAG Jazz

A jazz piano / music theory Q&A app grounded in 20 Wikipedia articles via retrieval-augmented generation.

Ask a question, get matching source passages, and (when configured) a natural-language answer from Groq's free tier or local Ollama. If the LLM is unavailable, the app falls back to retrieval-only.

## Articles

| Topic | File |
|---|---|
| Jazz Piano | `jazz_piano.json` |
| Jazz Harmony | `jazz_harmony.json` |
| Chord (Music) | `chord_music.json` |
| Chord Progression | `chord_progression.json` |
| ii-V-I Progression | `ii-v-i_progression.json` |
| Circle of Fifths | `circle_of_fifths.json` |
| Blues Scale | `blues_scale.json` |
| Bebop Scale | `bebop_scale.json` |
| Jazz Improvisation | `jazz_improvisation.json` |
| Voicing (Music) | `voicing_music.json` |
| Comping (Jazz) | `comping_jazz.json` |
| Stride Piano | `stride_piano.json` |
| Boogie-Woogie | `boogie-woogie.json` |
| Modal Jazz | `modal_jazz.json` |
| Twelve-Bar Blues | `twelve-bar_blues.json` |
| Tritone Substitution | `tritone_substitution.json` |
| Jazz Scale | `jazz_scale.json` |
| Pentatonic Scale | `pentatonic_scale.json` |
| Rhythm Changes | `rhythm_changes.json` |
| Piano Blues | `piano_blues.json` |

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

`chunks.json` is already committed. Rebuild chunks only if you change articles:

```bash
python build_chunks.py
python build_vectorstore.py
```

### Web app

```bash
copy .env.example .env   # optional: set GROQ_API_KEY or LLM_PROVIDER=ollama
uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000

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

- `download_articles.py` — Primary Wikipedia downloader
- `download_remaining.py` — MediaWiki REST fallback
- `rebuild_manifest.py` — Regenerates `articles/_manifest.json`
