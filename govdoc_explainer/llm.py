import litellm


def make_llm_chat_request(
    messages,
    model: str = "ollama/llama3",
    temperature: float | None = None,
    api_base: str | None = None,
):
    kwargs = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if api_base:
        # OpenAI-compatible local servers (llama.cpp's llama-server, LM Studio, vLLM, ...)
        kwargs["api_base"] = api_base
        kwargs["api_key"] = "none"  # local servers ignore the key, but the openai provider requires one
    try:
        response = litellm.completion(**kwargs)
        text_response = response.choices[0].message.content
        if text_response:
            return text_response
        print(response)
        return None
    except Exception as e:
        print(e)
        return None


def model_string_from_config(llm) -> str:
    service = llm.chat_service_name
    model_name = llm.chat_model_name

    if service == "openai":
        return model_name
    elif service == "openai-compatible":
        # any OpenAI-compatible endpoint, e.g. llama.cpp's llama-server
        return f"openai/{model_name}"
    elif service == "anthropic":
        return f"claude-{model_name}" if not model_name.startswith("claude") else model_name
    elif service == "ollama":
        return f"ollama/{model_name}"
    return f"{service}/{model_name}"
