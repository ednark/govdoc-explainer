# govdoc-explainer

Pre-process government documents into searchable, LLM-summarized static HTML. Extracts text from federal policy documents (PDF, HTML, DOCX, XLSX), generates perspective-based LLM summaries, and builds a static site with semantic + keyword search.

## Quick Start

### Install
```bash
uv sync --extra dev
npm install
```

### Configure
Edit files in `./config/`:
- `sources.csv` — source documents to process (Category, Standard, Url)
- `perspectives.csv` — roles for perspective-based summaries (Role, Prompt)
- `llm.txt` — LLM model selection (chat_service_name, chat_model_name)
- `prompts/*.txt` — prompt templates (overall, punchline, actions, keywords)

Optional: for local LLMs, start Ollama via `docker compose up -d`.

### Build
```bash
npm run build
```

This runs:
1. `python -m govdoc_explainer` — extracts text, generates embeddings + LLM summaries, renders HTML
2. `npx pagefind --site .` — builds the keyword search index

### Serve
```bash
npm run serve
```
Opens a local server with the generated site.

### Validate Source URLs
```bash
python scripts/validate_sources.py
```
HEAD-checks every URL in `sources.csv` for broken links and archived domains.

## Architecture

```
config/sources.csv → extract_text_from_url() → text
                                              ↓
                    generate_embeddings_for_text_sections() → embedding.json
                    generate_summaries_for_url() → summary files
                    generate_index_page_for_url() → index.html per source
                                              ↓
                    generate_main_embeddings() → assets/embedding.json
                    generate_lunr_index() → assets/lunr_index.json
                    generate_main_index_page() → index.html
                                              ↓
                    npx pagefind → assets/pagefind/ (keyword search index)
```

### Modules

- `govdoc_explainer/cli.py` — entry point, `process_sources()` pipeline
- `govdoc_explainer/config.py` — loads sources.csv, perspectives.csv, llm.txt, prompts/
- `govdoc_explainer/llm.py` — litellm wrapper (unified OpenAI/Anthropic/Ollama)
- `govdoc_explainer/extract.py` — URL → text (HTML/PDF/XLSX/DOCX)
- `govdoc_explainer/embeddings.py` — fastembed (all-MiniLM-L6-v2, 384-dim ONNX)
- `govdoc_explainer/summarize.py` — LLM summary generation
- `govdoc_explainer/render.py` — HTML page generation + lunr index
- `govdoc_explainer/text_utils.py` — chunking (TF-IDF similarity), name shortening

### Search

**Build-time:**
- `fastembed` generates `embedding.json` per source + aggregated `assets/embedding.json`
- Pagefind generates keyword search index in `assets/pagefind/`

**Client-side:**
- Transformers.js (Xenova/all-MiniLM-L6-v2) embeds the query and does cosine similarity against pre-generated vectors

The same model (all-MiniLM-L6-v2) is used on both sides for vector compatibility.

## Development

```bash
uv sync --extra dev          # install with dev deps
uv run ruff check govdoc_explainer/ tests/
uv run ruff format --check govdoc_explainer/ tests/
uv run pytest
```

## Optional: Local LLM via Ollama

```bash
docker compose up -d         # starts Ollama + Open WebUI
open http://localhost:3000/  # pull models (llama3.1, phi3, gemma2)
```

Then set in `config/llm.txt`:
```
chat_service_name: ollama
chat_model_name: llama3.1
```
