# govdoc-listener

A side-project "listener" that retrieves executive briefs and references from the
govdoc-explainer corpus. It is read-only over the main project's built artifacts
(`../sources/*/embedding.json`, summaries, `relevance.json`) and never modifies them.

## Setup

```bash
cd listener
uv sync --all-extras
```

## Run

```bash
cd listener
uv run govdoc-listener --port 8765
```

Then open http://127.0.0.1:8765 . Type a question and the dashboard returns the
most relevant documents with their executive brief, applicability/severity/urgency
badges, affected teams, and matching excerpts that link into the static doc pages.

Options:

- `--root <path>` — path to the govdoc-explainer project root (defaults to the parent of this project)
- `--host` / `--port` — bind address (defaults to 127.0.0.1:8765)

## Test

```bash
cd listener
uv run pytest
```

## How it works

- `corpus.py` — loads each source dir's chunk embeddings plus the cached
  executive brief, overall summary, keywords, and relevance JSON.
- `retrieve.py` — embeds the query with the same `all-MiniLM-L6-v2` model used at
  build time, ranks chunks by cosine similarity, and boosts scores with keyword
  matches and the document's applicability rating.
- `server.py` — FastAPI app: `/` dashboard, `/api/ask`, `/api/events` (SSE), and
  `/site/...` serving the main project's static pages and assets.

## Roadmap

- Phase 2: local microphone capture + Whisper transcription for in-person meetings.
- Phase 3: Slack Socket Mode integration to answer questions raised in channels.
