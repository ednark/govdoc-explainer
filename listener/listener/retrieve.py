from dataclasses import dataclass

import numpy as np
from fastembed import TextEmbedding

from .corpus import Document

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_embed_model():
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL_NAME)
    return _model


@dataclass
class ChunkHit:
    chunk_id: int
    text: str
    score: float
    url: str


@dataclass
class DocHit:
    label: str
    dir_name: str
    score: float
    relevance: dict | None
    exec_brief: str
    chunks: list[ChunkHit]


class Retriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._matrices: dict[str, np.ndarray] = {}
        for doc in documents:
            matrix = np.vstack([chunk.embedding for chunk in doc.chunks])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrices[doc.dir_name] = matrix / norms

    def embed_query(self, query: str) -> np.ndarray:
        vector = np.asarray(next(iter(get_embed_model().embed([query]))), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def search(self, query: str, max_docs: int = 5, max_chunks: int = 3) -> list[DocHit]:
        query_vector = self.embed_query(query)
        return self.search_vector(query_vector, query, max_docs=max_docs, max_chunks=max_chunks)

    def search_vector(
        self, query_vector: np.ndarray, query_text: str, max_docs: int = 5, max_chunks: int = 3
    ) -> list[DocHit]:
        query_lower = query_text.lower()
        scored: list[tuple[float, Document, list[ChunkHit]]] = []

        for doc in self.documents:
            matrix = self._matrices[doc.dir_name]
            similarities = matrix @ query_vector
            top_indices = np.argsort(similarities)[::-1][:max_chunks]
            best_score = float(similarities[top_indices[0]]) if len(top_indices) else 0.0

            keyword_hits = sum(1 for keyword in doc.keywords if keyword and keyword in query_lower)
            keyword_boost = min(keyword_hits, 5) * 0.02

            relevance_boost = 0.0
            if doc.relevance:
                relevance_boost = {"high": 0.05, "medium": 0.02}.get(
                    str(doc.relevance.get("applicability", "")).lower(), 0.0
                )

            chunk_hits = []
            for index in top_indices:
                score = float(similarities[index])
                if score <= 0.0:
                    continue
                chunk = doc.chunks[int(index)]
                chunk_hits.append(
                    ChunkHit(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=score,
                        url=f"/site/sources/{doc.dir_name}/index.html#chunk-{chunk.chunk_id}",
                    )
                )

            if not chunk_hits:
                continue

            total_score = best_score + keyword_boost + relevance_boost
            scored.append((total_score, doc, chunk_hits))

        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[DocHit] = []
        for total_score, doc, chunk_hits in scored[:max_docs]:
            results.append(
                DocHit(
                    label=doc.label,
                    dir_name=doc.dir_name,
                    score=round(total_score, 4),
                    relevance=doc.relevance,
                    exec_brief=doc.exec_brief,
                    chunks=chunk_hits,
                )
            )
        return results
