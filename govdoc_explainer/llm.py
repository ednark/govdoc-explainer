import litellm


def make_llm_chat_request(messages, model: str = "ollama/llama3"):
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
        )
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
    elif service == "anthropic":
        return f"claude-{model_name}" if not model_name.startswith("claude") else model_name
    elif service == "ollama":
        return f"ollama/{model_name}"
    return f"{service}/{model_name}"
