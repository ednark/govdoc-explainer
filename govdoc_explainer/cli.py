import argparse
import signal
from datetime import datetime

from govdoc_explainer.config import load_config
from govdoc_explainer.embeddings import generate_embeddings_for_url, generate_main_embeddings
from govdoc_explainer.extract import extract_text_from_url
from govdoc_explainer.render import generate_index_page_for_url, generate_lunr_index, generate_main_index_page
from govdoc_explainer.summarize import generate_summaries_for_url

_shutdown_requested = False


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _request_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def install_signal_handlers():
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)


def process_sources(config, only=""):
    sources_with_url = [(standard, source) for standard, source in config.sources.items() if source.url]

    filters = [f.strip().lower() for f in only.split(",") if f.strip()] if only else []
    if filters:
        sources_with_url = [
            (standard, source) for standard, source in sources_with_url if any(f in standard.lower() for f in filters)
        ]
        log(f"--only filter active: {len(sources_with_url)} source(s) match")

    total = len(sources_with_url)
    failed = []
    completed = 0

    for index, (standard, source) in enumerate(sources_with_url, start=1):
        if _shutdown_requested:
            log(f"Shutdown signal received — stopping before document {index}/{total}")
            log(f"Graceful shutdown: {completed}/{total} documents completed. Re-run the build to resume.")
            return
        url = source.url
        std = source.standard
        log(f"Processing {index}/{total}: {std}")
        try:
            extract_text_from_url(url, label=std)
            generate_embeddings_for_url(url, label=std)
            generate_summaries_for_url(url, label=std, config=config)
            generate_index_page_for_url(url, label=std, config=config)
            completed += 1
        except Exception as e:
            log(f"FAILED {index}/{total}: {std} — {type(e).__name__}: {e}")
            failed.append(std)

    generate_main_embeddings(config)
    generate_lunr_index(config)
    generate_main_index_page(config)

    log(f"Done: {total - len(failed)}/{total} sources processed successfully")
    if failed:
        log(f"Failed sources ({len(failed)}):")
        for std in failed:
            log(f"  - {std}")


def main():
    parser = argparse.ArgumentParser(description="govdoc-explainer build pipeline")
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated name substrings; only matching sources are processed",
    )
    args = parser.parse_args()
    install_signal_handlers()
    config = load_config("./config")
    process_sources(config, only=args.only)


if __name__ == "__main__":
    main()
