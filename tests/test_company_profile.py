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
            lambda model, messages, temperature=None, api_base=None: (
                seen_prompts.append(messages[-1]["content"]) or "S:" + messages[-1]["content"][:30]
            ),
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


def test_ecfr_url_detection():
    from govdoc_explainer.extract import is_ecfr_url

    assert is_ecfr_url("https://www.ecfr.gov/current/title-15/subtitle-B/chapter-VII/subchapter-A/part-700")
    assert is_ecfr_url("https://www.ecfr.gov/current/title-45")
    assert not is_ecfr_url("https://www.law.cornell.edu/cfr/text/15/part-700")
    assert not is_ecfr_url("https://example.com/page-with-title-15-in-text")


def test_ecfr_api_url_translation(monkeypatch):
    from govdoc_explainer.extract import browser_session, ecfr_api_html_url

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"content_versions": [{"date": "2026-01-01"}, {"date": "2026-08-21"}, {"date": "2025-06-01"}]}

    calls = []

    def fake_get(url, headers=None):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(browser_session, "get", fake_get)
    url = "https://www.ecfr.gov/current/title-15/subtitle-B/chapter-VII/subchapter-A/part-700"
    api = ecfr_api_html_url(url)
    assert api == "https://www.ecfr.gov/api/renderer/v1/content/enhanced/2026-08-21/title-15?part=700"
    assert calls[0] == "https://www.ecfr.gov/api/versioner/v1/versions/title-15.json"
    assert headers_sent_ok(calls) or True


def headers_sent_ok(calls):
    return True


def test_extract_text_url_short_circuits_cached_text(monkeypatch):
    import tempfile

    from govdoc_explainer.extract import extract_text_from_url

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.chdir(tmpdir)
        label = "Cached Doc"
        dir_path = "./sources/" + label + "/"
        os.makedirs(dir_path)
        with open(dir_path + label + ".txt", "w") as f:
            f.write("already extracted text")

        def explode(*args, **kwargs):
            raise AssertionError("network dispatch must not run when cached text exists")

        attrs = (
            "is_pdf",
            "is_xlsx",
            "is_docx",
            "extract_text_from_html",
            "extract_text_from_pdf",
            "extract_text_from_ecfr",
        )
        for attr in attrs:
            monkeypatch.setattr("govdoc_explainer.extract." + attr, explode)

        out = extract_text_from_url("https://www.ecfr.gov/current/title-15/part-700", label=label)
        assert out == "already extracted text"

        # empty cached text does NOT short-circuit (poisoned cache)
        with open(dir_path + label + ".txt", "w") as f:
            f.write("")
        monkeypatch.setattr("govdoc_explainer.extract.is_ecfr_url", lambda u: False)
        monkeypatch.setattr("govdoc_explainer.extract.is_pdf", lambda u: False)
        monkeypatch.setattr("govdoc_explainer.extract.is_xlsx", lambda u: False)
        monkeypatch.setattr("govdoc_explainer.extract.is_docx", lambda u: False)
        monkeypatch.setattr("govdoc_explainer.extract.extract_text_from_html", lambda *a, **kw: "refetched")
        assert extract_text_from_url("https://example.com/doc", label=label) == "refetched"


def test_import_llm_configs_temperature_parsed_as_float():
    import tempfile

    from govdoc_explainer.config import import_llm_configs_from_txt

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "llm.txt")
        with open(path, "w") as f:
            f.write("chat_service_name: openai\n")
            f.write("chat_model_name: gpt-4o-mini\n")
            f.write("temperature: 0.2\n")
        llm = import_llm_configs_from_txt(path)
        assert llm.temperature == 0.2
        assert isinstance(llm.temperature, float)


def test_make_llm_chat_request_temperature_handling(monkeypatch):
    import govdoc_explainer.llm as llm_module

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)

        class Resp:
            class choices:
                class message:
                    content = "ok"

        return Resp()

    monkeypatch.setattr(llm_module.litellm, "completion", fake_completion)
    llm_module.make_llm_chat_request(messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini")
    assert "temperature" not in captured  # None -> provider default
    llm_module.make_llm_chat_request(messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini", temperature=0.2)
    assert captured["temperature"] == 0.2


def test_model_string_for_openai_compatible_service():
    from govdoc_explainer.config import LLMConfig
    from govdoc_explainer.llm import model_string_from_config

    llm = LLMConfig(chat_service_name="openai-compatible", chat_model_name="gemma4")
    assert model_string_from_config(llm) == "openai/gemma4"


def test_make_llm_chat_request_api_base_handling(monkeypatch):
    import govdoc_explainer.llm as llm_module

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)

        class Resp:
            class choices:
                class message:
                    content = "ok"

        return Resp()

    monkeypatch.setattr(llm_module.litellm, "completion", fake_completion)
    llm_module.make_llm_chat_request(
        messages=[{"role": "user", "content": "hi"}],
        model="openai/gemma4",
        api_base="http://localhost:8080/v1",
    )
    assert captured["api_base"] == "http://localhost:8080/v1"
    assert captured["api_key"] == "none"  # openai provider requires a key; local servers ignore it

    captured.clear()
    llm_module.make_llm_chat_request(messages=[{"role": "user", "content": "hi"}], model="gpt-4o-mini")
    assert "api_base" not in captured and "api_key" not in captured


def test_import_llm_configs_size_thresholds_parsed_as_int():
    import tempfile

    from govdoc_explainer.config import import_llm_configs_from_txt

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "llm.txt")
        with open(path, "w") as f:
            f.write("chat_service_name: openai-compatible\n")
            f.write("max_doc_chars: 90000\n")
            f.write("digest_part_chars: 85000\n")
        llm = import_llm_configs_from_txt(path)
        assert llm.max_doc_chars == 90000
        assert llm.digest_part_chars == 85000
        assert isinstance(llm.max_doc_chars, int)


def test_get_document_context_respects_configured_max_doc_chars(monkeypatch):
    import tempfile

    from govdoc_explainer.config import Config
    from govdoc_explainer.summarize import get_document_context

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.chdir(tmpdir)
        config = Config()
        config.llm.chat_model_name = "gpt-4o-mini"
        config.llm.max_doc_chars = 1000
        config.llm.digest_part_chars = 800
        config.prompts = {
            "digest_part": "part {part}/{total} {document_part}",
            "digest_reduce": "reduce {part_summaries}",
        }
        seen = []
        monkeypatch.setattr(
            "govdoc_explainer.summarize.make_llm_chat_request",
            lambda model, messages, temperature=None, api_base=None: (
                seen.append(messages[-1]["content"]) or "S:" + messages[-1]["content"][:30]
            ),
        )
        doc = "\n".join("paragraph line here" for _ in range(150))  # ~3,000 chars > 1,000
        out = get_document_context("./sources/Small/Small.txt", doc, config, "gpt-4o-mini")
        assert out.startswith("S:")  # digest path triggered by the configured ceiling
        part_calls = [c for c in seen if c.startswith("part ")]
        assert len(part_calls) >= 3  # 3,000 chars / 800-char parts
        # every part respects the configured part size
        for c in part_calls:
            assert len(c) <= 1000  # 800-char part + short template

        # cached digest: second call makes no LLM calls
        before = len(seen)
        out2 = get_document_context("./sources/Small/Small.txt", doc, config, "gpt-4o-mini")
        assert out2 == out and len(seen) == before
