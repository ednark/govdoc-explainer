from govdoc_explainer.config import load_config
from govdoc_explainer.embeddings import generate_embeddings_for_url, generate_main_embeddings
from govdoc_explainer.extract import extract_text_from_url
from govdoc_explainer.render import generate_index_page_for_url, generate_lunr_index, generate_main_index_page
from govdoc_explainer.summarize import generate_summaries_for_url


def process_sources(config):
    for standard, source in config.sources.items():
        url = source.url
        std = source.standard
        if not url:
            continue
        print("Processing: " + std)
        extract_text_from_url(url, label=std)
        generate_embeddings_for_url(url, label=std)
        generate_summaries_for_url(url, label=std, config=config)
        generate_index_page_for_url(url, label=std, config=config)
    generate_main_embeddings(config)
    generate_lunr_index(config)
    generate_main_index_page(config)


def main():
    config = load_config("./config")
    process_sources(config)


if __name__ == "__main__":
    main()
