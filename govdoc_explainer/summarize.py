import hashlib
import json
import os
from pathlib import Path

from govdoc_explainer.extract import extract_text_from_url
from govdoc_explainer.llm import make_llm_chat_request, model_string_from_config
from govdoc_explainer.text_utils import fs_safe_url

MAX_DOC_CHARS = 400_000


def summary_artifact_path(text_file_path, config, prompt_name, prompt_text):
    """Path of an LLM-generated artifact, keyed by chat model and a hash of the rendered prompts.

    The prompt_text must be the exact text sent to the LLM (system + user prompt). Changing any
    input — company profile, perspectives, or a prompt template — produces a new hash, so the
    artifact regenerates instead of silently reusing stale output.
    """
    context_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:8]
    return text_file_path.replace(".txt", f".{config.llm.chat_model_name}.{context_hash}.summary.{prompt_name}.txt")


def artifact_manifest_path(text_file_path):
    return text_file_path.replace(".txt", ".artifacts.json")


def record_artifact(text_file_path, prompt_name, artifact_file_path):
    """Remember which generated file holds each prompt's output, so render can find them."""
    manifest_path = artifact_manifest_path(text_file_path)
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as file:
                manifest = json.load(file)
        except json.JSONDecodeError:
            manifest = {}
    manifest[prompt_name] = os.path.basename(artifact_file_path)
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)


def lookup_artifact(text_file_path, prompt_name):
    """Return the recorded artifact file for a prompt, if it exists on disk."""
    manifest_path = artifact_manifest_path(text_file_path)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r") as file:
            manifest = json.load(file)
    except json.JSONDecodeError:
        return None
    artifact_name = manifest.get(prompt_name)
    if not artifact_name:
        return None
    artifact_path = os.path.join(os.path.dirname(text_file_path), artifact_name)
    return artifact_path if os.path.exists(artifact_path) else None


def roles_block(config):
    """Rendered role list injected into prompts that need to name owner roles."""
    if not config.perspectives:
        return "- any team role relevant to the business described above"
    return "\n".join(
        f"- {p.role}: {p.description}" + (f" Interests: {p.interests}" if p.interests else "")
        for p in config.perspectives.values()
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

    system_prompt = config.prompts["system_context"].format(
        document_text=document_text, company_profile=config.company_profile
    )
    model = model_string_from_config(config.llm)

    perspectives = config.perspectives

    generate_overall_summary(system_prompt, text_file_path, config, model)
    generate_punchline_summary(system_prompt, text_file_path, perspectives, config, model)
    generate_action_summaries(system_prompt, text_file_path, perspectives, config, model)
    generate_keyword_summary(system_prompt, text_file_path, config, model)
    generate_executive_brief(text_file_path, document_text, config, model)


def _write_or_fail(summary_file_path, response, failure_message):
    if response:
        with open(summary_file_path, "w") as file:
            file.write(response)
        return response
    print(failure_message + " " + summary_file_path)
    return ""


def generate_overall_summary(system_prompt, text_file_path, config, model):
    prompt_name = "overall"
    user_prompt = config.prompts[prompt_name]
    summary_file_path = summary_artifact_path(text_file_path, config, prompt_name, system_prompt + user_prompt)
    record_artifact(text_file_path, prompt_name, summary_file_path)
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    print("Generating overall summary")
    response = make_llm_chat_request(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _write_or_fail(summary_file_path, response, "Failed to generate summary")


def generate_punchline_summary(system_prompt, text_file_path, perspectives, config, model):
    prompt_name = "punchline"
    user_prompt = config.prompts[prompt_name] + "\n- "
    user_prompt += "\n- ".join(f"{p.role}: {p.description}" for p in perspectives.values())
    summary_file_path = summary_artifact_path(text_file_path, config, prompt_name, system_prompt + user_prompt)
    record_artifact(text_file_path, prompt_name, summary_file_path)
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    print("Generating punchline summaries")
    response = make_llm_chat_request(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _write_or_fail(summary_file_path, response, "Failed to generate summary")


def generate_action_summaries(system_prompt, text_file_path, perspectives, config, model):
    for perspective, perspective_data in perspectives.items():
        generate_action_summary(system_prompt, text_file_path, perspective, perspective_data, config, model)


def generate_action_summary(system_prompt, text_file_path, perspective, perspective_data, config, model):
    prompt_name = f"actions.{perspective}"
    user_prompt = config.prompts["actions"]
    user_prompt += "\n Consider things from only this one perspective:"
    user_prompt += "\n" + perspective_data.prompt
    summary_file_path = summary_artifact_path(text_file_path, config, prompt_name, system_prompt + user_prompt)
    record_artifact(text_file_path, prompt_name, summary_file_path)
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    print("Generating action summary: " + perspective)
    response = make_llm_chat_request(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _write_or_fail(summary_file_path, response, "Failed to generate summary")


def generate_keyword_summary(system_prompt, text_file_path, config, model):
    prompt_name = "keywords"
    user_prompt = config.prompts[prompt_name]
    summary_file_path = summary_artifact_path(text_file_path, config, prompt_name, system_prompt + user_prompt)
    record_artifact(text_file_path, prompt_name, summary_file_path)
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    print("Generating keyword summary")
    response = make_llm_chat_request(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _write_or_fail(summary_file_path, response, "Failed to generate summary")


def generate_executive_brief(text_file_path, document_text, config, model):
    prompt_name = "exec_brief"
    system_message = "You are a clear, plain-language technical writer briefing company executives."
    user_prompt = config.prompts[prompt_name].format(
        document_text=document_text, company_profile=config.company_profile, roles=roles_block(config)
    )
    summary_file_path = summary_artifact_path(text_file_path, config, prompt_name, system_message + user_prompt)
    record_artifact(text_file_path, prompt_name, summary_file_path)
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r") as file:
            return file.read()
    print("Generating executive brief")
    response = make_llm_chat_request(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _write_or_fail(summary_file_path, response, "Failed to generate executive brief")
