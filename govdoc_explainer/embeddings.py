import base64
import json
import os
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from govdoc_explainer.extract import extract_text_from_url
from govdoc_explainer.text_utils import fs_safe_url, split_text_into_logical_sections

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
Q8_FORMAT = "govdoc-int8-v1"

# Metadata fields carried from fp32 entries into the q8 client files.
Q8_COPY_FIELDS = ("id", "title", "body", "keywords", "text")

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(EMBED_MODEL_NAME)
    return _embed_model


def generate_embeddings_for_text_sections(text):
    chunks = split_text_into_logical_sections(text, max_sentences_per_section=10, similarity_threshold=0.3)
    model = get_embed_model()
    embeddings = []
    chunk_texts = list(chunks)
    vectors = list(model.embed(chunk_texts))
    for chunk_id, (chunk, vector) in enumerate(zip(chunk_texts, vectors)):
        # L2-normalize so client-side dot products equal cosine similarity.
        # paraphrase-multilingual-MiniLM-L12-v2 ships without a Normalize
        # module, so fastembed returns vectors with norm ~2.7-4.7; storing
        # them raw reintroduces a length bias into ranking.
        v = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        embeddings.append(
            {
                "id": chunk_id,
                "text": chunk,
                "embedding": v.tolist(),
            }
        )
    return embeddings


def write_q8_file(entries, path):
    """Write the client-side int8 vector file next to the fp32 one.

    Symmetric per-vector int8 quantization; the client ranks on raw int8
    values (per-vector scale cancels in cosine similarity). ~4x smaller
    than fp32 JSON, benchmark-verified to preserve top-8 rankings.
    """
    out = []
    for entry in entries:
        new_entry = {k: entry[k] for k in Q8_COPY_FIELDS if k in entry}
        arr = np.asarray(entry["embedding"], dtype=np.float64)
        if arr.ndim == 2:
            arr = arr.reshape(-1)
        max_abs = float(np.max(np.abs(arr)))
        if max_abs == 0.0:
            scale, q = 1.0, np.zeros(EMBED_DIM, dtype=np.int8)
        else:
            scale = max_abs / 127.0
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        new_entry["embedding"] = base64.b64encode(q.tobytes()).decode("ascii")
        new_entry["scale"] = scale
        out.append(new_entry)
    with open(path, "w") as f:
        json.dump({"format": Q8_FORMAT, "dim": EMBED_DIM, "entries": out}, f)


def generate_embeddings_for_url(url, label=""):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    embed_file_path = dir_path + "/embedding.json"

    if os.path.exists(embed_file_path):
        return

    text = extract_text_from_url(url, label=label)
    if not text:
        return

    embedding = generate_embeddings_for_text_sections(text)
    with open(embed_file_path, "w") as f:
        json.dump(embedding, f)
    write_q8_file(embedding, dir_path + "/embedding_q8.json")


def generate_main_embeddings(config):
    print("Generating embedding for everything")
    main_embedding_file_path = "./assets/embedding.json"
    main_embeddings = []
    for standard, source in config.sources.items():
        url = source.url
        label = source.standard
        if not url:
            continue

        dir_path = "./sources/" + fs_safe_url(label) + "/"
        text_file_path = dir_path + fs_safe_url(label) + ".txt"
        embedding_file_path = dir_path + "embedding.json"

        standard_embeddings = []
        if os.path.exists(embedding_file_path):
            with open(embedding_file_path, "r") as file:
                file_embeddings = file.read()
                json_embeddings = json.loads(file_embeddings)
                if json_embeddings:
                    standard_embeddings = json_embeddings

        overall_summary = ""
        prompt_name = "overall"
        summary_file_path = text_file_path.replace(".txt", f".{config.llm.chat_model_name}.summary.{prompt_name}.txt")
        if os.path.exists(summary_file_path):
            with open(summary_file_path, "r") as file:
                overall_summary = file.read()

        keyword_summary = ""
        prompt_name = "keywords"
        summary_file_path = text_file_path.replace(".txt", f".{config.llm.chat_model_name}.summary.{prompt_name}.txt")
        if os.path.exists(summary_file_path):
            with open(summary_file_path, "r") as file:
                keyword_summary = file.read()

        safe_label = fs_safe_url(label)

        if not standard_embeddings and not keyword_summary and not overall_summary:
            continue

        overall_embedding = np.zeros(EMBED_DIM)
        section_count = 0
        for section in standard_embeddings:
            embedding = section["embedding"][0] if isinstance(section["embedding"][0], list) else section["embedding"]
            if isinstance(embedding, list) and len(embedding) == EMBED_DIM:
                overall_embedding += np.array(embedding)
                section_count += 1

        # Mean + L2-normalize so the client-side dot product equals cosine.
        # The previous raw sum made vector norms grow with document length
        # (up to ~515), biasing ranking toward chunk-heavy documents.
        if section_count:
            overall_embedding /= section_count
            norm = np.linalg.norm(overall_embedding)
            if norm > 0:
                overall_embedding /= norm

        main_embeddings.append(
            {
                "id": safe_label,
                "title": label,
                "body": overall_summary,
                "keywords": keyword_summary,
                "embedding": overall_embedding.tolist(),
            }
        )

    with open(main_embedding_file_path, "w") as f:
        json.dump(main_embeddings, f)
    write_q8_file(main_embeddings, "./assets/embedding_q8.json")
