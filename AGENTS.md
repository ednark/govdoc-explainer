# AGENTS.md — Build/lint/test commands for AI assistants

## Commands

### Install
```
uv sync
```

### Lint
```
ruff check govdoc_explainer/ tests/
ruff format --check govdoc_explainer/ tests/
```

### Tests
```
pytest
```

### Build (generate HTML + search index)
```
npm run build
```

### Validate source URLs
```
python scripts/validate_sources.py
```

### Serve locally
```
npm run serve
```

### Listener sub-project (separate from main build)
```
cd listener && uv sync --all-extras
cd listener && uv run pytest
cd listener && uv run govdoc-listener --port 8765
```
`listener/` is an independent uv project (FastAPI) that reads the main project's built `sources/` artifacts and serves a local retrieval dashboard at `http://127.0.0.1:8765`. It does not modify the main pipeline.

## Architecture

- `govdoc_explainer/cli.py` — entry point, `process_sources()` pipeline; `--only "name,substr"` limits a run to matching sources
- `govdoc_explainer/config.py` — loads sources.csv, sources-*.csv, manual docs dir, perspectives.csv, llm.txt, prompts/, company_profile.txt (falls back to company_profile_default.txt)
- `govdoc_explainer/llm.py` — litellm wrapper (unified OpenAI/Anthropic/Ollama)
- `govdoc_explainer/extract.py` — URL → text (HTML/PDF/XLSX/DOCX via requests + pymupdf + python-docx + openpyxl)
- `govdoc_explainer/embeddings.py` — fastembed (all-MiniLM-L6-v2, 384-dim ONNX)
- `govdoc_explainer/summarize.py` — LLM summary generation (overall/punchline/actions/keywords + executive brief + relevance JSON)
- `govdoc_explainer/render.py` — HTML page generation + lunr index
- `govdoc_explainer/text_utils.py` — chunking (TF-IDF similarity), name shortening (NLTK)

## Config

- `config/sources.csv` — source documents to process (Category, Standard, Url)
- `config/sources-*.csv` — extra source lists, merged into sources.csv (e.g. sources-de.csv)
- `sources/__manual-download-gov-docs/` — drop-in local files (PDF/XLSX/DOCX/HTML) auto-ingested as sources
- `config/perspectives.csv` — roles for perspective-based summaries (Role, Prompt)
- `config/llm.txt` — LLM model selection (chat_service_name, chat_model_name)
- `config/prompts/*.txt` — prompt templates
- `config/company_profile.txt` — user-local company context injected into the system_context + exec_brief prompts (gitignored; falls back to `company_profile_default.txt`); `company_profile_raw.txt` stores the pre-conversion description
- `config/perspectives.csv` — user-local review roles (Role, Description, Interests; gitignored; falls back to `perspectives_default.csv`) driving punchline + actions.<Role> summaries and the exec_brief role list
- LLM artifacts in `sources/<doc>/` are keyed by chat model + a hash of the rendered prompts (`summary_artifact_path`); a per-doc `.artifacts.json` manifest maps prompt names to artifact files for rendering. Note: `temperature` (llm.txt) is passed to the provider but is NOT part of the artifact hash — changing it does not invalidate the cache
- llm.txt services: openai, anthropic, ollama, and `openai-compatible` (any OpenAI-compatible endpoint via `chat_api_base`, e.g. llama.cpp's `llama-server`; litellm model string becomes `openai/<model_name>`)
- ecfr.gov `/current/` links are JavaScript SPAs — `extract.py` detects them and routes through eCFR's official API (versioner resolves the latest issue date, renderer serves static HTML scoped to the URL's part)

## Search

- **Build-time:** fastembed generates `embedding.json` per source + aggregated `assets/embedding.json`
- **Build-time:** Pagefind generates keyword search index in `assets/pagefind/`
- **Client-side:** Transformers.js (Xenova/all-MiniLM-L6-v2) embeds the query and does cosine similarity against pre-generated vectors

Same model (all-MiniLM-L6-v2) used both sides for vector compatibility.
