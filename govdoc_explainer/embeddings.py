import json
import os
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from govdoc_explainer.extract import extract_text_from_url
from govdoc_explainer.text_utils import fs_safe_url, split_text_into_logical_sections

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

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
        embeddings.append(
            {
                "id": chunk_id,
                "text": chunk,
                "embedding": vector.tolist(),
            }
        )
    return embeddings


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
        for section in standard_embeddings:
            embedding = section["embedding"][0] if isinstance(section["embedding"][0], list) else section["embedding"]
            if isinstance(embedding, list) and len(embedding) == EMBED_DIM:
                overall_embedding += np.array(embedding)

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
