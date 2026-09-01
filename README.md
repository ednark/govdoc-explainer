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
- `company_profile_default.txt` — the shipped example company profile
- `company_profile.txt` — your own company profile (optional; overrides the default)
- `prompts/*.txt` — prompt templates (overall, punchline, actions, keywords, exec_brief)

Optional: for local LLMs, start Ollama via `docker compose up -d`.

### Bring Your Own Company

Summaries, executive briefs, and relevance judgments are generated against a company profile. The shipped
`company_profile_default.txt` describes a federal IT contractor; to tailor the output to your organization:

1. Write a free-text description of your company to a file, e.g. `my_company.txt` (a few sentences to a few
   paragraphs: what you build, host, or operate; your stack; any compliance posture; team roles).
2. Convert it into a structured profile:
   ```bash
   npm run profile -- --from my_company.txt
   ```
   This saves your description to `config/company_profile_raw.txt`, generates a structured profile, shows it
   to you, and (after confirmation) writes `config/company_profile.txt`. Add `--yes` to skip the confirmation
   and `--force` to overwrite existing profile files.
3. Rebuild. Artifact filenames include a hash of the active profile and the chat model, so a new profile
   automatically regenerates every summary, brief, and relevance assessment on the next build:
   ```bash
   npm run build
   ```

`company_profile.txt` and `company_profile_raw.txt` are gitignored (they are yours); only the default is
tracked. To customize without the LLM step, copy the default to `company_profile.txt` and edit it by hand.

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
