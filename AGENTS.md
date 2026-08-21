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

## Architecture

- `govdoc_explainer/cli.py` — entry point, `process_sources()` pipeline
- `govdoc_explainer/config.py` — loads sources.csv, perspectives.csv, llm.txt, prompts/
- `govdoc_explainer/llm.py` — litellm wrapper (unified OpenAI/Anthropic/Ollama)
- `govdoc_explainer/extract.py` — URL → text (HTML/PDF/XLSX/DOCX via requests + fitz + python-docx + openpyxl)
- `govdoc_explainer/embeddings.py` — fastembed (all-MiniLM-L6-v2, 384-dim ONNX)
- `govdoc_explainer/summarize.py` — LLM summary generation (overall/punchline/actions/keywords)
- `govdoc_explainer/render.py` — HTML page generation + lunr index
- `govdoc_explainer/text_utils.py` — chunking (TF-IDF similarity), name shortening (NLTK)

## Config

- `config/sources.csv` — source documents to process (Category, Standard, Url)
- `config/perspectives.csv` — roles for perspective-based summaries (Role, Prompt)
- `config/llm.txt` — LLM model selection (chat_service_name, chat_model_name)
- `config/prompts/*.txt` — prompt templates

## Search

- **Build-time:** fastembed generates `embedding.json` per source + aggregated `assets/embedding.json`
- **Build-time:** Pagefind generates keyword search index in `assets/pagefind/`
- **Client-side:** Transformers.js (Xenova/all-MiniLM-L6-v2) embeds the query and does cosine similarity against pre-generated vectors

Same model (all-MiniLM-L6-v2) used both sides for vector compatibility.
