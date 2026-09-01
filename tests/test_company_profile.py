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


def test_summary_artifact_path_includes_model_and_profile_hash():
    from govdoc_explainer.config import Config
    from govdoc_explainer.summarize import summary_artifact_path

    config = Config()
    config.llm.chat_model_name = "gpt-4o-mini"
    config.company_profile = "We build federal websites."

    path = summary_artifact_path("./sources/Doc A/Doc A.txt", config, "overall")
    assert path.startswith("./sources/Doc A/Doc A.gpt-4o-mini.")
    assert path.endswith(".summary.overall.txt")
    assert config.profile_hash in path

    other = Config()
    other.llm.chat_model_name = "gpt-4o-mini"
    other.company_profile = "We build hospital networks."
    assert summary_artifact_path("./sources/Doc A/Doc A.txt", other, "overall") != path


def test_render_uses_same_artifact_path_as_summarize(monkeypatch):
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
        brief_path = summary_artifact_path(dir_path + label + ".txt", config, "exec_brief")
        with open(brief_path, "w") as f:
            f.write("## Executive Takeaway\nThis document matters.")
        html = executive_brief_html(label, config)
        assert "This document matters." in html
