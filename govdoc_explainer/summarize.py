import os
from pathlib import Path

from govdoc_explainer.extract import extract_text_from_url
from govdoc_explainer.llm import make_llm_chat_request, model_string_from_config
from govdoc_explainer.text_utils import fs_safe_url

MAX_DOC_CHARS = 400_000


def summary_artifact_path(text_file_path, config, prompt_name):
    """Path of an LLM-generated artifact, keyed by chat model and company profile hash.

    Changing either regenerates the artifact instead of silently reusing the old one.
    """
    return text_file_path.replace(
        ".txt", f".{config.llm.chat_model_name}.{config.profile_hash}.summary.{prompt_name}.txt"
    )


def generate_summaries_for_url(url, label, config):
    document_text = extract_text_from_url(url, label=label)
    if not document_text:
        return

    if len(document_text) > MAX_DOC_CHARS:
        print(f"Document too large for the LLM context window ({len(document_text)} chars), truncating")
        document_text = document_text[:MAX_DOC_CHARS]

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    system_prompt = config.prompts["system_context"].format(document_text=document_text)
    model = model_string_from_config(config.llm)

    perspectives = config.perspectives

    generate_overall_summary(system_prompt, text_file_path, config, model)
    generate_punchline_summary(system_prompt, text_file_path, perspectives, config, model)
    generate_action_summaries(system_prompt, text_file_path, perspectives, config, model)
    generate_keyword_summary(system_prompt, text_file_path, config, model)
    generate_executive_brief(text_file_path, document_text, config, model)


def generate_overall_summary(system_prompt, text_file_path, config, model):
    prompt_name = "overall"
    summary_file_path = summary_artifact_path(text_file_path, config, prompt_name)
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    print("Generating overall summary")
    user_prompt = config.prompts[prompt_name]
    response = make_llm_chat_request(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if response:
        with open(summary_file_path, "w") as file:
            file.write(response)
        return response
    print("Failed to generate summary " + summary_file_path)
    return ""


def generate_punchline_summary(system_prompt, text_file_path, perspectives, config, model):
    prompt_name = "punchline"
    summary_file_path = summary_artifact_path(text_file_path, config, prompt_name)
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    user_prompt = config.prompts[prompt_name] + "\n- "
    user_prompt += "\n- ".join(p.prompt for p in perspectives.values())
    print("Generating punchline summaries")
    response = make_llm_chat_request(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if response:
        with open(summary_file_path, "w") as file:
            file.write(response)
        return response
    print("Failed to generate summary " + summary_file_path)
    return ""


def generate_action_summaries(system_prompt, text_file_path, perspectives, config, model):
    for perspective, perspective_data in perspectives.items():
        generate_action_summary(system_prompt, text_file_path, perspective, perspective_data.prompt, config, model)


def generate_action_summary(system_prompt, text_file_path, perspective, perspective_prompt, config, model):
    prompt_name = "actions"
    summary_file_path = summary_artifact_path(text_file_path, config, f"{prompt_name}.{perspective}")
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    user_prompt = config.prompts[prompt_name]
    user_prompt += "\n Consider things from only this one perspective:"
    user_prompt += "\n" + perspective_prompt
    print("Generating action summary: " + perspective)
    response = make_llm_chat_request(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if response:
        with open(summary_file_path, "w") as file:
            file.write(response)
        return response
    print("Failed to generate summary " + summary_file_path)
    return ""


def generate_keyword_summary(system_prompt, text_file_path, config, model):
    prompt_name = "keywords"
    summary_file_path = summary_artifact_path(text_file_path, config, prompt_name)
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    user_prompt = config.prompts[prompt_name]
    print("Generating keyword summary")
    response = make_llm_chat_request(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if response:
        with open(summary_file_path, "w") as file:
            file.write(response)
        return response
    print("Failed to generate summary " + summary_file_path)
    return ""


def generate_executive_brief(text_file_path, document_text, config, model):
    prompt_name = "exec_brief"
    summary_file_path = summary_artifact_path(text_file_path, config, prompt_name)
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    print("Generating executive brief")
    user_prompt = config.prompts[prompt_name].format(
        document_text=document_text, company_profile=config.company_profile
    )
    response = make_llm_chat_request(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a clear, plain-language technical writer briefing company executives.",
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    if response:
        with open(summary_file_path, "w") as file:
            file.write(response)
        return response
    print("Failed to generate executive brief " + summary_file_path)
    return ""
