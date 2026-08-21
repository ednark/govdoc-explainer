import json
import os
import re
from pathlib import Path

import markdown2
from lunr import lunr

from govdoc_explainer.text_utils import fs_safe_url, split_text_into_logical_sections


def generate_index_page_for_url(url, label, config):
    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    index_file_path = dir_path + "/index.html"

    text_file_url = "./" + fs_safe_url(label) + ".txt"
    text_file_path = dir_path + "/" + fs_safe_url(label) + ".txt"

    text = ""
    if os.path.exists(text_file_path):
        with open(text_file_path, "r") as file:
            text = file.read()
    chunks = split_text_into_logical_sections(text, max_sentences_per_section=10, similarity_threshold=0.3)
    text_chunks = ""
    for chunk_id, chunk in enumerate(chunks):
        text_chunks += f"""<div id="chunk-{chunk_id}" class="text-chunk" /><a name="chunk-{chunk_id}"><sup>[{chunk_id}]</sup></a> {chunk}</div>"""

    summaries_html = ""

    prompts = {
        "overall": config.prompts["overall"],
        "punchline": config.prompts["punchline"]
        + "\n- "
        + "\n- ".join(p.prompt for p in config.perspectives.values()),
    }
    for perspective, perspective_data in config.perspectives.items():
        prompt_name = "actions." + perspective
        user_prompt = config.prompts["actions"]
        user_prompt += "\n Consider things from only this one perspective:"
        user_prompt += "\n" + perspective_data.prompt
        prompts[prompt_name] = user_prompt

    for prompt_name, _prompt in prompts.items():
        summary_file_path = text_file_path.replace(
            ".txt", f".{config.llm.chat_model_name}.summary.{prompt_name}.txt"
        )
        if os.path.exists(summary_file_path):
            summary_file_text = ""
            with open(summary_file_path, "r") as file:
                summary_file_text = file.read()
            summary_html = markdown2.markdown(summary_file_text)
            summary_title = prompt_name.title()
            summaries_html += f"""
                <div class="accordion">
                    <div class="accordion-item">
                        <button class="accordion-header">{summary_title} Summary</button>
                        <div class="accordion-content">{summary_html}</div>
                    </div>
                </div>
                <br /><br />
            """

    menu_html = """
        <div id="nav-menu" class="accordion" role="navigation" aria-label="Page Navigation">
            <div class="accordion-item">
                <button id="nav-menu-toggle" class="accordion-header"><span class="accordion-header-text">Standards</span><span class="accordion-header-icon"></span></button>
                <div class="accordion-content"><ul id="nav-menu-standards"></ul></div>
            </div>
        </div>
        <br /><br />
    """

    index_tmpl = f"""<html>
    <head>
        <link rel="stylesheet" type="text/css" href="../../assets/standards.css" />
        <script src="../../assets/standards.js" type="text/javascript"></script>

        <script src="../../assets/page_sources.js" type="text/javascript"></script>
        <script src="../../assets/nav.js" type="text/javascript"></script>
        <script type="module" src="../../assets/semantic_search.js"></script>
    </head>
    <body>
        <h1>{label}</h1>

        {menu_html}

        <div class="accordion">
            <div class="accordion-item">
                <button class="accordion-header" id="source-data-button">Source Data</button>
                <div class="accordion-content" id="source-data-content">
                    <br />
                    <a href="{url}">Raw Data</a> | <a href="{text_file_url}">Source Text</a>
                    <br /><br />
                    <div id="embed-query">
                        <input type="text" id="embed-query-input" placeholder="Semantic search..."/>
                        <button id="embed-query-button">Search</button>
                        <span id="embed-query-message"></span>
                    </div>
                    <br /><br />
                    <div class="embed-search-results">{text_chunks}</div>
                </div>
            </div>
        </div>
        <br /><br />

        {summaries_html}

    </body>
    </html>"""

    with open(index_file_path, "w") as file:
        file.write(index_tmpl)


def generate_main_index_page(config):
    index_file_path = "./index.html"

    prompts = {
        "overall": config.prompts["overall"],
        "punchline": config.prompts["punchline"],
        "keywords": config.prompts["keywords"],
    }
    for perspective, perspective_data in config.perspectives.items():
        prompt_name = "actions." + perspective
        user_prompt = config.prompts["actions"]
        user_prompt += "\n Consider things from only this one perspective:"
        user_prompt += "\n" + perspective_data.prompt
        prompts[prompt_name] = user_prompt
        prompts["punchline"] += "\n- " + perspective

    prompts_html = ""
    for prompt_name, prompt in prompts.items():
        prompt = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", prompt)
        prompt_html = markdown2.markdown(prompt)
        prompt_title = prompt_name.title()
        prompts_html += f"""
            <div class="accordion">
                <div class="accordion-item">
                    <button class="accordion-header">{prompt_title} Prompt</button>
                    <div class="accordion-content">{prompt_html}</div>
                </div>
            </div>
            <br /><br />
        """

    sources_js = {}
    for standard, source in config.sources.items():
        url = str(source.url)
        if not url:
            continue
        standard_index_file_path = "./sources/" + fs_safe_url(standard) + "/index.html"
        sources_js[standard] = standard_index_file_path

    sources_js = json.dumps(sources_js)
    with open("./assets/sources.js", "w") as file:
        file.write(f"var sources = {sources_js};")

    page_sources_js = {}
    for standard, source in config.sources.items():
        url = str(source.url)
        if not url:
            continue
        standard_index_file_path = "../" + fs_safe_url(standard) + "/index.html"
        page_sources_js[standard] = standard_index_file_path

    page_sources_js = json.dumps(page_sources_js)
    with open("./assets/page_sources.js", "w") as file:
        file.write(f"var sources = {page_sources_js};")

    menu_html = """
        <div id="nav-menu" class="accordion" role="navigation" aria-label="Page Navigation">
            <div class="accordion-item">
                <button id="nav-menu-toggle" class="accordion-header"><span class="accordion-header-text">Standards</span><span class="accordion-header-icon"></span></button>
                <div class="accordion-content"><ul id="nav-menu-standards"></ul></div>
            </div>
        </div>
        <br /><br />
    """

    search_html = """
        <div id="embed-query">
            <input type="text" id="embed-query-input" placeholder="Search across all standards..."/>
            <button id="embed-query-button">Search</button>
            <span id="embed-query-message"></span>
        </div>
        <br /><br />
    """

    index_tmpl = f"""<html>
        <head>
        <link rel="stylesheet" type="text/css" href="./assets/standards.css" />
        <script src="./assets/standards.js"></script>

        <script src="./assets/sources.js"></script>
        <script src="./assets/nav.js"></script>
        <script type="module" src="./assets/semantic_search.js"></script>
        </head>
    <body>
        <h1>Gov Doc Summaries</h1>

        {search_html}

        {prompts_html}

        {menu_html}

    </body>
    </html>"""

    with open(index_file_path, "w") as file:
        file.write(index_tmpl)


def generate_lunr_index(config):
    print("Generating search index for everything")
    search_documents = []
    for standard, source in config.sources.items():
        url = source.url
        label = source.standard
        if not url:
            continue

        dir_path = "./sources/" + fs_safe_url(label) + "/"
        text_file_path = dir_path + fs_safe_url(label) + ".txt"

        overall_summary = ""
        prompt_name = "overall"
        summary_file_path = text_file_path.replace(
            ".txt", f".{config.llm.chat_model_name}.summary.{prompt_name}.txt"
        )
        if os.path.exists(summary_file_path):
            with open(summary_file_path, "r") as file:
                overall_summary = file.read()

        keyword_summary = ""
        prompt_name = "keywords"
        summary_file_path = text_file_path.replace(
            ".txt", f".{config.llm.chat_model_name}.summary.{prompt_name}.txt"
        )
        if os.path.exists(summary_file_path):
            with open(summary_file_path, "r") as file:
                keyword_summary = file.read()

        safe_label = fs_safe_url(label)

        if not overall_summary and not keyword_summary:
            continue

        search_documents.append({
            "id": safe_label,
            "title": label,
            "body": overall_summary,
            "keywords": keyword_summary,
        })

    index = lunr(
        ref="id",
        fields=["title", "body", "keywords"],
        documents=search_documents,
    )
    index_data = index.serialize()
    with open("./assets/lunr_index.json", "w") as file:
        json.dump(index_data, file)
