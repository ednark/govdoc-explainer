from govdoc_explainer.config import LLMConfig
from govdoc_explainer.llm import model_string_from_config


def test_model_string_openai():
    llm = LLMConfig(chat_service_name="openai", chat_model_name="gpt-4o-mini")
    assert model_string_from_config(llm) == "gpt-4o-mini"


def test_model_string_ollama():
    llm = LLMConfig(chat_service_name="ollama", chat_model_name="llama3.1")
    assert model_string_from_config(llm) == "ollama/llama3.1"


def test_model_string_anthropic():
    llm = LLMConfig(chat_service_name="anthropic", chat_model_name="claude-sonnet-4-20250514")
    assert model_string_from_config(llm) == "claude-sonnet-4-20250514"
