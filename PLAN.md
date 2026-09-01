# govdoc-explainer Modernization Plan

## Goals

Modernize the codebase while keeping it a **CLI tool** that outputs static files:
- Reproducible dependency management
- Clean, modular, tested Python code
- Simplified multi-provider LLM via `litellm`
- Deduplicated, lighter frontend assets
- Proper documentation
- No web framework; no USWDS; keep client-side TF.js search

## Current State Summary

| Area | Now | Problems |
|------|-----|----------|
| Deps | None (manual `pip install`) | Not reproducible |
| `.gitignore` | Missing | `node_modules/`, `.venv/`, `sources/`, `ollama/` pollute repo |
| Entry point | `extract_content.py` (1184 lines) | Monolithic, global mutable `config` dict |
| LLM | Custom OpenAI/Anthropic/Ollama wrappers | Stale model names, duplicated logic |
| Embeddings (py) | TF Hub USE + BERT, loaded twice at import | Heavy, unused BERT/spaCy/NLTK paths |
| Embeddings (JS) | TF.js + USE (1.4MB), loaded client-side | Works but duplicated across 2 files |
| Search JS | `embedding_search.js` + `page_embedding_search.js` | ~95% identical, ~1000 lines each |
| Docker | Empty `Dockerfile`, partial `docker-compose.yml` | Broken |
| Tests / CI | None | — |
| Docs | Minimal README | — |

---

## Phase 1 — Repo Hygiene & Packaging

**Goal:** Make the project reproducible and clean.

### 1.1 Add `.gitignore`
```
.venv/
node_modules/
ollama/
sources/
__pycache__/
*.pyc
.DS_Store
# Keep config/ tracked, keep assets/ tracked (generated output)
```
Note: `sources/` contains 77 processed doc dirs (~large). Decide whether to keep a few samples tracked or gitignore entirely. Recommend gitignoring `sources/` and documenting how to regenerate.

### 1.2 Add `pyproject.toml` (PEP 621)
- Use `setuptools` or `hatchling` backend
- Project metadata (name, version, description, python-requires)
- Dependencies:
  - `requests`, `beautifulsoup4`, `pymupdf` (fitz), `python-docx`, `openpyxl`, `pandas`
  - `litellm` (replaces openai + anthropic clients)
  - `tensorflow`, `tensorflow-hub` (for USE embeddings)
  - `scikit-learn` (TF-IDF chunking)
  - `nltk` (sentence tokenization for chunking — keep, it's used)
  - `markdown2`, `lunr`
  - Drop `spacy`, `torch`, `transformers` (unused paths — BERT/keyword/shorten via spacy are dead code)
- Optional `[project.optional-dependencies]` dev group: `pytest`, `ruff`
- `[project.scripts]` entry point: `govdoc = govdoc_explainer.cli:main`
- `[tool.ruff]` config

### 1.3 Remove dead code & unused deps
- Delete `shorten_standard_name_via_llm` (unused — nltk version is used)
- Delete `generate_embeddings_for_text_sections_via_bert` and `generate_keyword_summary_via_bert` / `_via_spacy` (commented-out call sites)
- Delete the BERT model loading at lines 1176-1180 (`embed_model_bert`, `embed_tokenizer_bert`, `keyword_model`, `keyword_tokenizer`)
- Delete `nlp = spacy.load("en_core_web_sm")` and `import spacy`
- Delete duplicate `embed_model = hub.load(...)` at line 38 (keep the one at line 1171, or better — load once in a function)
- Delete `import tensorflow as tf` and `import tensorflow_hub as hub` duplicates (lines 30-31 vs 38)
- Remove `flexsearch` from package.json (only `lunr` is used in JS; flexsearch is a stale dep)
- ~~Decide fate of `config/solicitation_analyzer.gpt`, `config/default_questions.txt`, `config/resumes.txt`~~ DONE 2026-09-01: removed from repo and history (legacy prompts from a previous employer, not wired into the pipeline).

### 1.4 Package structure
```
govdoc-explainer/
  pyproject.toml
  README.md
  PLAN.md
  .gitignore
  config/              # user-editable (tracked)
  assets/              # generated + static JS/CSS (tracked)
  govdoc_explainer/    # the python package (new)
    __init__.py
    cli.py             # entry point, argparse
    config.py          # load config/ (sources, perspectives, prompts, llm)
    llm.py             # litellm wrapper
    extract.py         # URL → text (html/pdf/xlsx/docx)
    embeddings.py     # TF Hub USE embeddings
    summarize.py       # generate overall/punchline/actions/keywords
    search_index.py    # lunr index + embedding.json generation
    render.py          # HTML page generation (templating)
    text_utils.py      # chunking, TF-IDF similarity, shorten_standard_name
    __main__.py        # python -m govdoc_explainer
  sources/             # generated output (gitignored)
  index.html           # generated (gitignored or tracked)
  docker-compose.yml   # ollama only
```

---

## Phase 2 — LLM Modernization (litellm)

**Goal:** Replace custom OpenAI/Anthropic/Ollama wrappers with `litellm`.

### 2.1 Replace `make_llm_chat_request`
`litellm.completion()` accepts a unified `model` string:
- OpenAI: `"gpt-4o"`, `"gpt-4o-mini"`
- Anthropic: `"claude-3-5-sonnet-20241022"` (or latest)
- Ollama: `"ollama/llama3"`

Single function replaces ~50 lines of provider branching.

### 2.2 Update `config/llm.txt` format
```
# Model selection (litellm model string)
chat_model: gpt-4o-mini

# Embedding model (kept separate, still TF Hub USE)
embed_model: universal-sentence-encoder/4
```
Remove the `chat_service_name` / `chat_model_name` split — litellm handles routing via the model string.

### 2.3 Simplify summary caching
Current: filename includes model name (`{model}.summary.{prompt}.txt`). Keep this pattern (good for cache invalidation when switching models) but use the single `chat_model` string.

---

## Phase 3 — Frontend Cleanup

**Goal:** Keep client-side TF.js search, but deduplicate and lighten.

### 3.1 Deduplicate search JS
- Extract shared code (stemmer, metaphone, stopwords, cosine similarity, hybrid similarity) into `assets/search_utils.js`
- `embedding_search.js` (main index page) and `page_embedding_search.js` (per-source page) become thin wrappers that import shared utils and implement only their page-specific result rendering
- Reduces ~2000 lines to ~500

### 3.2 Consolidate TF.js loading
- Keep `tf.js` (1.4MB) and `tf-universal-sentence-encoder.js` but ensure they're loaded once
- Consider adding `defer` and a loading indicator (currently the page just hangs while TF.js loads the USE model)

### 3.3 Clean up HTML generation
- Replace f-string HTML in `render.py` with Python's `string.Template` or simple Jinja2-free templating
- Fix the missing `<head>` tag in generated `index.html` (currently `<html>` → `<link>` → `<body>` with no `<head>`)
- Keep the accordion CSS/JS as-is (per user preference: keep custom CSS)

### 3.4 Fix `generate_main_index_page`
- Currently writes `sources.js` and `page_sources.js` with different relative paths (`./sources/...` vs `../...`) — keep but make explicit
- The `prompts["punchline"]` mutation at line 1070 appends perspective names in a confusing way — clean up

---

## Phase 4 — Code Quality

### 4.1 Remove global mutable state
- `config` dict is mutated by `import_configs` and read everywhere
- Pass `config` as a parameter to functions, or use a `Config` dataclass loaded once in `cli.py`

### 4.2 Fix model loading at import time
- `embed_model = hub.load(...)` runs on `import` — move into a lazy loader or `main()`
- Same for NLTK `word_freq = FreqDist(nltk_words.words())`

### 4.3 Type hints & docstrings
- Add type hints to all public functions
- Add docstrings to modules and key functions

### 4.4 Linting
- Add `ruff` config in `pyproject.toml`
- Run `ruff check` and `ruff format`
- Add `AGENTS.md` noting `ruff check` as the lint command

### 4.5 Tests
- `tests/test_extract.py` — test text extraction from sample HTML/PDF (use small fixture files)
- `tests/test_config.py` — test config loading from CSV/txt
- `tests/test_text_utils.py` — test chunking, shorten_standard_name
- `tests/test_llm.py` — mock litellm.completion, test summary generation
- Use `pytest` with fixtures in `tests/fixtures/`
- Target: cover the pure-logic functions (config parsing, text chunking, name shortening, HTML rendering) — skip network/LLM calls or mock them

---

## Phase 5 — Documentation

### 5.1 README.md (rewrite)
- Project description & purpose
- Quick start (install, configure, run)
- Architecture overview (pipeline diagram in ASCII)
- Configuration reference (`config/` files explained)
- How to add a new source / perspective / prompt
- How search works (TF.js + USE + hybrid phonetic matching)
- Development setup (venv, ruff, pytest)

### 5.2 `CONTRIBUTING.md`
- Code style (ruff)
- How to run tests
- How to add features (new document types, new LLM providers)

### 5.3 `AGENTS.md`
- Build/lint/test commands for AI assistants

### 5.4 Inline docs
- Module-level docstrings explaining each stage of the pipeline

---

## Phase 6 — Docker & Infra

### 6.1 Fix or remove `Dockerfile`
Currently empty. Options:
- **Remove** if the CLI is meant to run locally (recommended — the Python deps include TensorFlow which is painful in Docker)
- **Or** write a working `Dockerfile` that installs deps and runs the pipeline

### 6.2 Keep `docker-compose.yml`
- Ollama + Open WebUI compose is fine for local LLM dev
- Add a comment that this is optional (only needed for local models)

---

## Implementation Order

| Step | Phase | Effort | Risk |
|------|-------|--------|------|
| 1 | 1.1 `.gitignore` | Trivial | None |
| 2 | 1.2 `pyproject.toml` | Small | None |
| 3 | 1.3 Remove dead code | Small | Low (verify nothing breaks) |
| 4 | 1.4 Package structure | Medium | Medium (move code, update imports) |
| 5 | 2.1-2.3 litellm | Medium | Low (well-tested lib) |
| 6 | 4.1-4.2 Remove globals, lazy loading | Small | Low |
| 7 | 3.1 Deduplicate JS | Medium | Low (mechanical) |
| 8 | 3.3-3.4 HTML fixes | Small | Low |
| 9 | 4.4-4.5 Lint + tests | Medium | None |
| 10 | 5.1-5.4 Docs | Medium | None |
| 11 | 6.1 Docker | Small | None |

**Recommended approach:** Steps 1-3 first (safe, immediate value), then 4-5 (the big refactor), then 6-8 (cleanup), then 9-11 (quality + docs). Each step should be a separate commit.

---

## Open Questions

1. **`sources/` directory (77 processed docs, ~large):** Gitignore and document regeneration, or keep tracked?
2. ~~**Legacy employer files** (`solicitation_analyzer.gpt`, `default_questions.txt`, `resumes.txt`)~~ RESOLVED 2026-09-01: removed from repo and history.
3. **TensorFlow dependency:** It's heavy (~500MB installed). Keep for embeddings, or switch to `sentence-transformers` (lighter) or OpenAI embeddings API? (Plan currently keeps TF per user preference for client-side TF.js consistency.)
4. **Python version target:** 3.11+ recommended (litellm and modern typing). Acceptable?
