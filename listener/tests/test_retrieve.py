import json

import numpy as np

from listener.corpus import EMBED_DIM, load_corpus
from listener.retrieve import Retriever


def _vec(components):
    """Build a 384-dim unit-ish vector with the given leading components."""
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    for i, c in enumerate(components):
        v[i] = c
    return v.tolist()


def _make_project(tmp_path):
    sources = tmp_path / "sources"

    doc_a = sources / "Doc A"
    doc_a.mkdir(parents=True)
    (doc_a / "Doc A.embedding.json").write_text("[]")  # wrong name, should be ignored
    (doc_a / "embedding.json").write_text(
        json.dumps(
            [
                {"id": 0, "text": "alpha content", "embedding": _vec([1.0, 0.0, 0.0])},
                {"id": 1, "text": "more alpha", "embedding": _vec([0.9, 0.1, 0.0])},
            ]
        )
    )
    (doc_a / "Doc A.gpt-4o-mini.summary.exec_brief.txt").write_text("Brief A")
    (doc_a / "Doc A.gpt-4o-mini.summary.keywords.txt").write_text("alpha\naccessibility\n")
    (doc_a / "Doc A.gpt-4o-mini.relevance.json").write_text(
        json.dumps({"applicability": "high", "severity": "medium", "urgency": "low", "reason": "r"})
    )

    doc_b = sources / "Doc B"
    doc_b.mkdir(parents=True)
    (doc_b / "embedding.json").write_text(
        json.dumps([{"id": 0, "text": "beta content", "embedding": _vec([0.0, 1.0, 0.0])}])
    )

    empty_dir = sources / "No Embeddings"
    empty_dir.mkdir(parents=True)
    (empty_dir / "notes.txt").write_text("nothing")

    return tmp_path


def test_load_corpus_reads_docs_and_metadata(tmp_path):
    root = _make_project(tmp_path)
    docs = load_corpus(root, model_name="gpt-4o-mini")
    assert [d.label for d in docs] == ["Doc A", "Doc B"]

    doc_a = docs[0]
    assert len(doc_a.chunks) == 2
    assert doc_a.exec_brief == "Brief A"
    assert doc_a.keywords == ["alpha", "accessibility"]
    assert doc_a.relevance["applicability"] == "high"


def test_load_corpus_missing_sources_dir(tmp_path):
    assert load_corpus(tmp_path / "nowhere") == []


def test_retriever_ranks_by_similarity_and_boosts(tmp_path):
    root = _make_project(tmp_path)
    docs = load_corpus(root, model_name="gpt-4o-mini")
    retriever = Retriever(docs)

    query_vector = np.zeros(EMBED_DIM, dtype=np.float32)
    query_vector[0] = 1.0
    results = retriever.search_vector(query_vector, "tell me about alpha accessibility")

    assert results[0].label == "Doc A"
    assert results[0].chunks[0].text == "alpha content"
    assert results[0].relevance["applicability"] == "high"
    assert results[0].chunks[0].url == "/site/sources/Doc A/index.html#chunk-0"

    labels = [r.label for r in results]
    assert "Doc B" not in labels


def test_retriever_skips_zero_similarity_docs(tmp_path):
    root = _make_project(tmp_path)
    docs = load_corpus(root, model_name="gpt-4o-mini")
    retriever = Retriever(docs)

    query_vector = np.zeros(EMBED_DIM, dtype=np.float32)
    query_vector[2] = 1.0
    results = retriever.search_vector(query_vector, "unrelated question")
    assert results == []
