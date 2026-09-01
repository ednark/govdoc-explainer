import json
import os

from govdoc_explainer.config import load_config
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
