import json
import os

from govdoc_explainer.config import Config, LLMConfig, Source, load_config
from govdoc_explainer.render import priority_briefing_html, relevance_badges_html
from govdoc_explainer.summarize import parse_relevance_json


def test_parse_relevance_json_valid():
    response = json.dumps(
        {
            "applicability": "high",
            "severity": "medium",
            "urgency": "low",
            "affected_teams": ["Developer", "Security"],
            "reason": "Applies to our federal websites",
        }
    )
    result = parse_relevance_json(response)
    assert result["applicability"] == "high"
    assert result["severity"] == "medium"
    assert result["urgency"] == "low"
    assert result["affected_teams"] == ["Developer", "Security"]
    assert result["reason"] == "Applies to our federal websites"


def test_parse_relevance_json_fenced():
    response = '```json\n{"applicability": "medium", "severity": "high", "urgency": "high"}\n```'
    result = parse_relevance_json(response)
    assert result["applicability"] == "medium"
    assert result["severity"] == "high"


def test_parse_relevance_json_with_surrounding_text():
    response = 'Here is the assessment: {"applicability": "low", "severity": "low", "urgency": "low"} Hope that helps.'
    result = parse_relevance_json(response)
    assert result is not None
    assert result["applicability"] == "low"


def test_parse_relevance_json_garbage():
    assert parse_relevance_json("no json here at all") is None
    assert parse_relevance_json("") is None
    assert parse_relevance_json(None) is None


def test_parse_relevance_json_defaults():
    result = parse_relevance_json('{"applicability": "bogus", "severity": "bogus", "affected_teams": "not a list"}')
    assert result["applicability"] == "low"
    assert result["severity"] == "low"
    assert result["urgency"] == "low"
    assert result["affected_teams"] == []


def test_load_config_company_profile():
    import csv
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "sources.csv"), "w", newline="") as f:
            writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerow(["Category", "Standard", "Url"])
        with open(os.path.join(tmpdir, "company_profile.txt"), "w") as f:
            f.write("We build federal websites.")
        config = load_config(tmpdir)
        assert config.company_profile == "We build federal websites."


def _make_config():
    config = Config()
    config.llm = LLMConfig(chat_service_name="openai", chat_model_name="gpt-4o-mini")
    config.sources = {
        "Doc A": Source(category="Cat", standard="Doc A", title="Doc A", url="https://example.com/a"),
        "Doc B": Source(category="Cat", standard="Doc B", title="Doc B", url="https://example.com/b"),
        "Doc C": Source(category="Cat", standard="Doc C", title="Doc C", url="https://example.com/c"),
    }
    return config


def _write_relevance(label, model, data):
    dir_path = "./sources/" + label + "/"
    os.makedirs(dir_path, exist_ok=True)
    with open(dir_path + label + "." + model + ".relevance.json", "w") as f:
        json.dump(data, f)


def test_priority_briefing_ranks_and_filters(monkeypatch):
    with monkeypatch.context() as m:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            m.chdir(tmpdir)
            config = _make_config()
            _write_relevance(
                "Doc A",
                "gpt-4o-mini",
                {
                    "applicability": "medium",
                    "severity": "high",
                    "urgency": "low",
                    "affected_teams": ["Security"],
                    "reason": "rA",
                },
            )
            _write_relevance(
                "Doc B",
                "gpt-4o-mini",
                {
                    "applicability": "high",
                    "severity": "high",
                    "urgency": "high",
                    "affected_teams": ["Developer"],
                    "reason": "rB",
                },
            )
            _write_relevance(
                "Doc C",
                "gpt-4o-mini",
                {"applicability": "low", "severity": "high", "urgency": "high", "affected_teams": [], "reason": "rC"},
            )

            html = priority_briefing_html(config)
            assert "Priority Briefing" in html
            assert html.index("Doc B") < html.index("Doc A")
            assert "Doc C" not in html


def test_priority_briefing_empty_when_no_relevances(monkeypatch):
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.chdir(tmpdir)
        config = _make_config()
        assert priority_briefing_html(config) == ""


def test_relevance_badges_html():
    html = relevance_badges_html({"applicability": "high", "severity": "low", "urgency": "medium", "reason": "because"})
    assert "badge-high" in html
    assert "Applies: high" in html
    assert "because" in html
    assert relevance_badges_html(None) == ""
