import argparse
import os
import signal
import sys
from datetime import datetime

from govdoc_explainer.config import (
    COMPANY_PROFILE_FILENAME,
    COMPANY_PROFILE_RAW_FILENAME,
    load_config,
)
from govdoc_explainer.embeddings import generate_embeddings_for_url, generate_main_embeddings
from govdoc_explainer.extract import extract_text_from_url
from govdoc_explainer.llm import make_llm_chat_request, model_string_from_config
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


def run_build(args):
    install_signal_handlers()
    config = load_config("./config")
    if not config.company_profile:
        log(
            "WARNING: no company profile found (config/company_profile.txt or company_profile_default.txt); "
            "LLM prompts will run without company context"
        )
    process_sources(config, only=args.only)


def run_profile(args):
    if not os.path.exists(args.from_file):
        print(f"Description file not found: {args.from_file}")
        sys.exit(1)
    with open(args.from_file, "r") as file:
        description = file.read().strip()
    if not description:
        print(f"Description file is empty: {args.from_file}")
        sys.exit(1)

    config = load_config("./config")
    for target in (COMPANY_PROFILE_RAW_FILENAME, COMPANY_PROFILE_FILENAME):
        target_path = os.path.join("./config", target)
        if os.path.exists(target_path) and not args.force:
            print(f"{target_path} already exists; re-run with --force to overwrite it")
            sys.exit(1)

    prompt = config.prompts.get("company_profile")
    if not prompt:
        print("Missing prompt template: config/prompts/company_profile.txt")
        sys.exit(1)

    raw_path = os.path.join("./config", COMPANY_PROFILE_RAW_FILENAME)
    with open(raw_path, "w") as file:
        file.write(description)
    log(f"Saved raw description to {raw_path}")

    log(f"Generating company profile with {config.llm.chat_service_name}/{config.llm.chat_model_name}...")
    response = make_llm_chat_request(
        model=model_string_from_config(config.llm),
        messages=[
            {
                "role": "system",
                "content": "You convert company descriptions into structured company profiles. "
                "You follow the requested output structure exactly and never invent facts.",
            },
            {"role": "user", "content": prompt.format(description=description)},
        ],
    )
    if not response or not response.strip():
        print("LLM returned no usable profile; nothing written.")
        sys.exit(1)

    print("\n--- Generated company profile ---\n")
    print(response.strip())
    print("\n---------------------------------\n")

    profile_path = os.path.join("./config", COMPANY_PROFILE_FILENAME)
    if not args.yes:
        answer = input(f"Write this profile to {profile_path}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted; profile not written. The raw description was kept at " + raw_path)
            return

    with open(profile_path, "w") as file:
        file.write(response.strip() + "\n")
    log(f"Wrote {profile_path}")
    log("Re-run the build to regenerate all LLM summaries against the new profile.")


def main():
    parser = argparse.ArgumentParser(description="govdoc-explainer pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="run the full build pipeline")
    build_parser.add_argument(
        "--only",
        default="",
        help="comma-separated name substrings; only matching sources are processed",
    )
    build_parser.set_defaults(func=run_build)

    profile_parser = subparsers.add_parser(
        "profile",
        help="convert a free-text company description into config/company_profile.txt",
    )
    profile_parser.add_argument(
        "--from", dest="from_file", required=True, help="path to a plain-text company description"
    )
    profile_parser.add_argument("--yes", action="store_true", help="write the profile without asking for confirmation")
    profile_parser.add_argument("--force", action="store_true", help="overwrite existing profile files")
    profile_parser.set_defaults(func=run_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
