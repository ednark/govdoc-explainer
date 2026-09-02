import csv
import hashlib
import os
from dataclasses import dataclass, field

MANUAL_SOURCES_DIRNAME = "__manual-download-gov-docs"
LOCAL_SOURCE_EXTENSIONS = {".pdf", ".xlsx", ".docx", ".html", ".htm"}

COMPANY_PROFILE_FILENAME = "company_profile.txt"
COMPANY_PROFILE_DEFAULT_FILENAME = "company_profile_default.txt"
COMPANY_PROFILE_RAW_FILENAME = "company_profile_raw.txt"
PERSPECTIVES_FILENAME = "perspectives.csv"
PERSPECTIVES_DEFAULT_FILENAME = "perspectives_default.csv"


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
    description: str
    interests: str = ""

    @property
    def prompt(self) -> str:
        """Full perspective prompt: who the reviewer is plus what they care about."""
        if self.interests:
            return f"You are {self.description} You care especially about: {self.interests}."
        return f"You are {self.description}"


@dataclass
class LLMConfig:
    chat_service_name: str = "ollama"
    chat_model_name: str = "llama3"
    embed_model_name: str = ""
    keyword_model_name: str = ""
    temperature: float | None = None  # None = provider default
    chat_api_base: str = ""  # only used by the openai-compatible service


@dataclass
class Config:
    sources: dict[str, Source] = field(default_factory=dict)
    perspectives: dict[str, Perspective] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    llm: LLMConfig = field(default_factory=LLMConfig)
    company_profile: str = ""
    company_profile_source: str = ""
    perspectives_source: str = ""

    @property
    def profile_hash(self) -> str:
        """Short hash of the active company profile; part of LLM artifact cache keys."""
        if not self.company_profile:
            return "noprf"
        return hashlib.sha256(self.company_profile.encode("utf-8")).hexdigest()[:8]


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
                key = key.strip()
                value = value.strip()
                if key == "temperature":
                    llm.temperature = float(value)
                else:
                    setattr(llm, key, value)
    return llm


def import_perspectives_from_csv(file_path) -> dict[str, Perspective]:
    perspectives = {}
    for row in read_csv_skip_empty(file_path):
        if row[0].startswith("#"):
            continue  # commented-out perspective row
        if len(row) >= 3:
            role, description, interests = row[0], row[1], row[2]
        elif len(row) == 2:
            role, description, interests = row[0], row[1], ""
        else:
            continue
        perspectives[role] = Perspective(role=role, description=description, interests=interests)
    return perspectives


def import_sources_from_csv(file_path) -> dict[str, Source]:
    sources = {}
    for row in read_csv_skip_empty(file_path):
        if len(row) >= 3:
            category, standard, url = row
            sources[standard] = Source(category=category, standard=standard, title=standard, url=url)
    return sources


def import_sources_from_local_dir(dir_path) -> dict[str, Source]:
    sources = {}
    if not os.path.isdir(dir_path):
        return sources
    for file_name in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file_name)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in LOCAL_SOURCE_EXTENSIONS:
            continue
        standard = os.path.splitext(file_name)[0]
        sources[standard] = Source(
            category="Manually Downloaded",
            standard=standard,
            title=standard,
            url=file_path,
        )
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
        if file_name == "llm.txt":
            config.llm = import_llm_configs_from_txt(file_path)

    # user-local profile wins; shipped default is the fallback
    if os.path.exists(dir_path + COMPANY_PROFILE_FILENAME):
        config.company_profile_source = COMPANY_PROFILE_FILENAME
    elif os.path.exists(dir_path + COMPANY_PROFILE_DEFAULT_FILENAME):
        config.company_profile_source = COMPANY_PROFILE_DEFAULT_FILENAME
    if config.company_profile_source:
        with open(dir_path + config.company_profile_source, "r") as file:
            config.company_profile = file.read()

    # user-local perspectives win; shipped default is the fallback
    if os.path.exists(dir_path + PERSPECTIVES_FILENAME):
        config.perspectives_source = PERSPECTIVES_FILENAME
    elif os.path.exists(dir_path + PERSPECTIVES_DEFAULT_FILENAME):
        config.perspectives_source = PERSPECTIVES_DEFAULT_FILENAME
    if config.perspectives_source:
        config.perspectives = import_perspectives_from_csv(dir_path + config.perspectives_source)

    prompts_dir = dir_path + "prompts"
    if os.path.isdir(prompts_dir):
        config.prompts = import_config_prompts(prompts_dir)

    sources_file = dir_path + "sources.csv"
    if os.path.exists(sources_file):
        config.sources = import_sources_from_csv(sources_file)

    for file_name in sorted(os.listdir(dir_path)):
        if file_name.startswith("sources") and file_name.endswith(".csv") and file_name != "sources.csv":
            config.sources.update(import_sources_from_csv(dir_path + file_name))

    project_root = os.path.dirname(os.path.abspath(dir_path))
    manual_dir = os.path.join(project_root, "sources", MANUAL_SOURCES_DIRNAME)
    config.sources.update(import_sources_from_local_dir(manual_dir))

    return config
