import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EMBED_DIM = 384
DEFAULT_MODEL_NAME = "gpt-4o-mini"


@dataclass
class Chunk:
    chunk_id: int
    text: str
    embedding: np.ndarray


@dataclass
class Document:
    label: str
    dir_name: str
    chunks: list[Chunk] = field(default_factory=list)
    exec_brief: str = ""
    overall_summary: str = ""
    relevance: dict | None = None
    keywords: list[str] = field(default_factory=list)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_first_suffix(dir_path: Path, suffix: str) -> str:
    for path in sorted(dir_path.iterdir()):
        if path.is_file() and path.name.endswith(suffix):
            return path.read_text(errors="ignore")
    return ""


def load_document(dir_path: Path, model_name: str) -> Document | None:
    embed_file = dir_path / "embedding.json"
    if not embed_file.exists():
        return None

    try:
        raw_chunks = json.loads(embed_file.read_text(errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return None

    chunks: list[Chunk] = []
    for raw in raw_chunks:
        if not isinstance(raw, dict):
            continue
        embedding = raw.get("embedding")
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            embedding = embedding[0]
        if not isinstance(embedding, list) or len(embedding) != EMBED_DIM:
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=int(raw.get("id", len(chunks))),
                text=text,
                embedding=np.asarray(embedding, dtype=np.float32),
            )
        )

    if not chunks:
        return None

    doc = Document(label=dir_path.name, dir_name=dir_path.name, chunks=chunks)
    doc.exec_brief = _read_first_suffix(dir_path, f".{model_name}.summary.exec_brief.txt")
    doc.overall_summary = _read_first_suffix(dir_path, f".{model_name}.summary.overall.txt")
    keywords_text = _read_first_suffix(dir_path, f".{model_name}.summary.keywords.txt")
    doc.keywords = [line.strip().lower() for line in keywords_text.splitlines() if line.strip()]

    relevance_text = _read_first_suffix(dir_path, f".{model_name}.relevance.json")
    if relevance_text:
        try:
            relevance = json.loads(relevance_text)
            if isinstance(relevance, dict):
                doc.relevance = relevance
        except json.JSONDecodeError:
            doc.relevance = None

    return doc


def load_corpus(project_root: str | Path | None = None, model_name: str = DEFAULT_MODEL_NAME) -> list[Document]:
    root = Path(project_root) if project_root else default_project_root()
    sources_dir = root / "sources"
    documents: list[Document] = []
    if not sources_dir.is_dir():
        return documents

    for entry in sorted(os.listdir(sources_dir)):
        dir_path = sources_dir / entry
        if not dir_path.is_dir():
            continue
        doc = load_document(dir_path, model_name)
        if doc is not None:
            documents.append(doc)
    return documents
