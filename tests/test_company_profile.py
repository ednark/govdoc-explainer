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


def test_split_document_respects_max_chars_and_breaks_at_lines():
    from govdoc_explainer.summarize import split_document

    small = "a" * 100
    assert split_document(small, 200) == [small]

    text = "\n".join(f"line {i} " + "x" * 50 for i in range(400))
    parts = split_document(text, 3000)
    assert len(parts) > 1
    assert all(len(p) <= 3000 for p in parts)
    assert "\n".join(parts).replace("\n\n", "\n").startswith("line 0")


def test_get_document_context_small_doc_passthrough(monkeypatch):
    import tempfile

    from govdoc_explainer.config import Config
    from govdoc_explainer.summarize import get_document_context

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.chdir(tmpdir)
        config = Config()
        calls = []
        monkeypatch.setattr("govdoc_explainer.summarize.make_llm_chat_request", lambda **kw: calls.append(kw) or "")
        doc = "small document " * 10
        out = get_document_context("./sources/D/D.txt", doc, config, "gpt-4o-mini")
        assert out == doc
        assert calls == []


def test_get_document_context_maps_and_reduces_oversized_doc(monkeypatch):
    import tempfile

    from govdoc_explainer.config import Config
    from govdoc_explainer.summarize import get_document_context

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.chdir(tmpdir)
        config = Config()
        config.llm.chat_model_name = "gpt-4o-mini"
        config.company_profile = "We build federal websites."
        config.prompts = {
            "digest_part": "part {part}/{total} {company_profile} {document_part}",
            "digest_reduce": "reduce: {company_profile} {part_summaries}",
        }
        seen_prompts = []
        monkeypatch.setattr(
            "govdoc_explainer.summarize.make_llm_chat_request",
            lambda model, messages: seen_prompts.append(messages[-1]["content"]) or "S:" + messages[-1]["content"][:30],
        )
        doc = "\n".join("paragraph line " for _ in range(40000))  # > 300k chars
        out = get_document_context("./sources/Big/Big.txt", doc, config, "gpt-4o-mini")
        assert out.startswith("S:")  # one part-summary call per part + one reduce call
        part_calls = [p for p in seen_prompts if p.startswith("part ")]
        reduce_calls = [p for p in seen_prompts if p.startswith("reduce:")]
        assert len(part_calls) == 3 and len(reduce_calls) == 1
        assert len(part_calls[0]) <= 301_000  # each part respects the size cap

        # second call hits the digest cache: no new LLM calls
        out2 = get_document_context("./sources/Big/Big.txt", doc, config, "gpt-4o-mini")
        assert out2 == out
        assert len(seen_prompts) == 4
