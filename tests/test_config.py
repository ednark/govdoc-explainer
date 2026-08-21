import csv
import os
import tempfile

from govdoc_explainer.config import (
    import_llm_configs_from_txt,
    import_perspectives_from_csv,
    import_sources_from_csv,
    load_config,
)


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def test_import_sources_from_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sources.csv")
        _write_csv(
            path,
            ["Category", "Standard", "Url"],
            [
                ["Websites", "Test Standard", "https://example.com"],
                ["Security", "Another Standard", "https://example.gov"],
            ],
        )
        sources = import_sources_from_csv(path)
        assert len(sources) == 2
        assert "Test Standard" in sources
        assert sources["Test Standard"].url == "https://example.com"
        assert sources["Test Standard"].category == "Websites"


def test_import_sources_skips_empty_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sources.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerow(["Category", "Standard", "Url"])
            writer.writerow(["Cat", "Std", "https://example.com"])
            writer.writerow(["", "", ""])
            writer.writerow(["Cat2", "Std2", "https://example.gov"])
        sources = import_sources_from_csv(path)
        assert len(sources) == 2


def test_import_perspectives_from_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "perspectives.csv")
        _write_csv(
            path,
            ["Role", "Prompt"],
            [
                ["Developer", "Consider from a developer perspective"],
                ["Designer", "Consider from a designer perspective"],
            ],
        )
        perspectives = import_perspectives_from_csv(path)
        assert len(perspectives) == 2
        assert "Developer" in perspectives
        assert perspectives["Developer"].prompt == "Consider from a developer perspective"


def test_import_llm_configs_from_txt():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "llm.txt")
        with open(path, "w") as f:
            f.write("# comment line\n")
            f.write("chat_service_name: openai\n")
            f.write("chat_model_name: gpt-4o-mini\n")
        llm = import_llm_configs_from_txt(path)
        assert llm.chat_service_name == "openai"
        assert llm.chat_model_name == "gpt-4o-mini"


def test_load_config_full():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_csv(
            os.path.join(tmpdir, "sources.csv"),
            ["Category", "Standard", "Url"],
            [["Cat", "Std", "https://example.com"]],
        )
        _write_csv(
            os.path.join(tmpdir, "perspectives.csv"),
            ["Role", "Prompt"],
            [["Dev", "dev perspective"]],
        )
        with open(os.path.join(tmpdir, "llm.txt"), "w") as f:
            f.write("chat_service_name: openai\nchat_model_name: gpt-4o-mini\n")
        prompts_dir = os.path.join(tmpdir, "prompts")
        os.makedirs(prompts_dir)
        with open(os.path.join(prompts_dir, "overall.txt"), "w") as f:
            f.write("Summarize this document")

        config = load_config(tmpdir)
        assert len(config.sources) == 1
        assert len(config.perspectives) == 1
        assert config.llm.chat_model_name == "gpt-4o-mini"
        assert "overall" in config.prompts
