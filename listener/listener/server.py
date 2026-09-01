import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .corpus import default_project_root, load_corpus
from .retrieve import Retriever

_subscribers: list[asyncio.Queue] = []


def create_app(project_root: str | Path | None = None) -> FastAPI:
    root = Path(project_root) if project_root else default_project_root()
    documents = load_corpus(root)
    retriever = Retriever(documents)

    app = FastAPI(title="govdoc-listener")
    app.state.retriever = retriever
    app.state.root = root

    sources_dir = root / "sources"
    assets_dir = root / "assets"
    if sources_dir.is_dir():
        app.mount("/site/sources", StaticFiles(directory=str(sources_dir), html=True), name="sources")
    if assets_dir.is_dir():
        app.mount("/site/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return (Path(__file__).parent / "static" / "dashboard.html").read_text()

    @app.get("/site/index.html")
    def main_site_index():
        return FileResponse(str(root / "index.html"))

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "documents": len(documents)}

    @app.post("/api/ask")
    async def ask(request: Request):
        body = await request.json()
        question = (body.get("question") or "").strip()
        if not question:
            return JSONResponse({"error": "empty question"}, status_code=400)
        hits = retriever.search(question)
        payload = {"question": question, "results": [asdict(hit) for hit in hits]}
        for queue in list(_subscribers):
            queue.put_nowait(payload)
        return payload

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        queue: asyncio.Queue = asyncio.Queue()
        _subscribers.append(queue)

        async def stream():
            try:
                while True:
                    payload = await queue.get()
                    yield f"data: {json.dumps(payload)}\n\n"
            finally:
                _subscribers.remove(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="govdoc listener dashboard")
    parser.add_argument("--root", default=None, help="path to the govdoc-explainer project root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    import uvicorn

    app = create_app(args.root)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
