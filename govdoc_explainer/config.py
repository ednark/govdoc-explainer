import csv
import os
from dataclasses import dataclass, field


@dataclass
class Source:
    category: str
    standard: str
    title: str
    url: str

    @property
    def short_standard(self) -> str:
        if len(self.standard) > 100:
            from govdoc_explainer.text_utils import shorten_standard_name
            return shorten_standard_name(self.standard)
        return self.standard


@dataclass
class Perspective:
    role: str
    prompt: str


@dataclass
class LLMConfig:
    chat_service_name: str = "ollama"
    chat_model_name: str = "llama3"
    embed_model_name: str = ""
    keyword_model_name: str = ""


@dataclass
class Config:
    sources: dict[str, Source] = field(default_factory=dict)
    perspectives: dict[str, Perspective] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    llm: LLMConfig = field(default_factory=LLMConfig)


def read_csv_skip_empty(file_path):
    with open(file_path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)
        for row in reader:
            cleaned_row = [field.strip().strip('"') for field in row]
            if any(field for field in cleaned_row):
                yield cleaned_row


def import_llm_configs_from_txt(file_path) -> LLMConfig:
    llm = LLMConfig()
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split(":", 1)
                setattr(llm, key.strip(), value.strip())
    return llm


def import_perspectives_from_csv(file_path) -> dict[str, Perspective]:
    perspectives = {}
    for row in read_csv_skip_empty(file_path):
        if len(row) >= 2:
            role, prompt = row
            perspectives[role] = Perspective(role=role, prompt=prompt)
    return perspectives


def import_sources_from_csv(file_path) -> dict[str, Source]:
    sources = {}
    for row in read_csv_skip_empty(file_path):
        if len(row) >= 3:
            category, standard, url = row
            sources[standard] = Source(category=category, standard=standard, title=standard, url=url)
    return sources


def import_config_prompts(dir_path) -> dict[str, str]:
    prompts = {}
    for file_name in os.listdir(dir_path):
        file_path = dir_path + "/" + file_name
        with open(file_path, "r") as file:
            prompt_name = file_name.replace(".txt", "")
            prompts[prompt_name] = file.read()
    return prompts


def load_config(dir_path) -> Config:
    if not dir_path.endswith("/"):
        dir_path = dir_path + "/"

    config = Config()

    for file_name in os.listdir(dir_path):
        file_path = dir_path + file_name
        if file_name == "perspectives.csv":
            config.perspectives = import_perspectives_from_csv(file_path)
        elif file_name == "llm.txt":
            config.llm = import_llm_configs_from_txt(file_path)

    prompts_dir = dir_path + "prompts"
    if os.path.isdir(prompts_dir):
        config.prompts = import_config_prompts(prompts_dir)

    sources_file = dir_path + "sources.csv"
    if os.path.exists(sources_file):
        config.sources = import_sources_from_csv(sources_file)

    return config
