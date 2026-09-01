import csv
import os

from govdoc_explainer.config import load_config


def _write_minimal_config(tmpdir):
    with open(os.path.join(tmpdir, "sources.csv"), "w", newline="") as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["Category", "Standard", "Url"])


def test_load_config_company_profile_user_file_wins():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_minimal_config(tmpdir)
        with open(os.path.join(tmpdir, "company_profile_default.txt"), "w") as f:
            f.write("Default profile.")
        with open(os.path.join(tmpdir, "company_profile.txt"), "w") as f:
            f.write("User profile.")
        config = load_config(tmpdir)
        assert config.company_profile == "User profile."
        assert config.company_profile_source == "company_profile.txt"


def test_load_config_company_profile_falls_back_to_default():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_minimal_config(tmpdir)
        with open(os.path.join(tmpdir, "company_profile_default.txt"), "w") as f:
            f.write("Default profile.")
        config = load_config(tmpdir)
        assert config.company_profile == "Default profile."
        assert config.company_profile_source == "company_profile_default.txt"


def test_load_config_company_profile_missing_is_empty():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_minimal_config(tmpdir)
        config = load_config(tmpdir)
        assert config.company_profile == ""
        assert config.company_profile_source == ""


def test_profile_hash_depends_on_profile_content():
    from govdoc_explainer.config import Config

    config = Config()
    config.company_profile = "We build federal websites."
    first = config.profile_hash
    config.company_profile = "We build hospital networks."
    second = config.profile_hash
    assert first != second
    assert len(first) == 8

    config.company_profile = ""
    assert config.profile_hash == "noprf"


def test_summary_artifact_path_hashes_prompt_text():
    from govdoc_explainer.config import Config
    from govdoc_explainer.summarize import summary_artifact_path

    config = Config()
    config.llm.chat_model_name = "gpt-4o-mini"
    base = "./sources/Doc A/Doc A.txt"
    first = summary_artifact_path(base, config, "overall", "system prompt A" + "user prompt A")
    second = summary_artifact_path(base, config, "overall", "system prompt A" + "user prompt A")
    changed = summary_artifact_path(base, config, "overall", "system prompt B" + "user prompt A")
    assert first == second
    assert first != changed
    assert config.llm.chat_model_name in first
    assert ".summary.overall.txt" in first


def test_record_and_lookup_artifact(monkeypatch):
    import tempfile

    from govdoc_explainer.summarize import lookup_artifact, record_artifact

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.chdir(tmpdir)
        text_file_path = "./sources/Doc A/Doc A.txt"
        os.makedirs("./sources/Doc A")
        assert lookup_artifact(text_file_path, "overall") is None

        artifact = "./sources/Doc A/Doc A.gpt-4o-mini.ab12cd34.summary.overall.txt"
        with open(artifact, "w") as f:
            f.write("summary text")
        record_artifact(text_file_path, "overall", artifact)
        assert lookup_artifact(text_file_path, "overall") == artifact

        # recorded but deleted -> None
        os.remove(artifact)
        assert lookup_artifact(text_file_path, "overall") is None


def test_render_uses_artifact_manifest_for_exec_brief(monkeypatch):
    import json
    import tempfile

    from govdoc_explainer.config import Config, LLMConfig
    from govdoc_explainer.render import executive_brief_html
    from govdoc_explainer.summarize import summary_artifact_path

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.chdir(tmpdir)
        label = "Doc A"
        dir_path = "./sources/" + label + "/"
        os.makedirs(dir_path)
        config = Config()
        config.llm = LLMConfig(chat_service_name="openai", chat_model_name="gpt-4o-mini")
        config.company_profile = "We build federal websites."
        brief_path = summary_artifact_path(dir_path + label + ".txt", config, "exec_brief", "system" + "user")
        with open(brief_path, "w") as f:
            f.write("## Executive Takeaway\nThis document matters.")
        manifest_path = dir_path + label + ".artifacts.json"
        with open(manifest_path, "w") as f:
            json.dump({"exec_brief": os.path.basename(brief_path)}, f)
        html = executive_brief_html(label, config)
        assert "This document matters." in html

        # no manifest -> no brief
        os.remove(manifest_path)
        assert executive_brief_html(label, config) == ""
