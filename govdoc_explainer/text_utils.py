import re

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_words
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_word_freq = None


def get_word_freq():
    global _word_freq
    if _word_freq is None:
        _word_freq = FreqDist(nltk_words.words())
    return _word_freq


def fs_safe_url(url):
    return url.replace("/", "_").replace(":", "_")


def shorten_standard_name(standard):
    return shorten_standard_name_via_nltk(standard)


def shorten_standard_name_via_nltk(standard, min_length=200):
    words = word_tokenize(standard)

    tagged_words = pos_tag(words)

    stop_words = set(stopwords.words("english"))
    stop_words.update(
        ["federal", "government", "office", "agency", "Memorandum", "requirement", "requirements", "policy", "policies"]
    )

    word_freq = get_word_freq()

    word_importance = []
    for word, tag in tagged_words:
        score = 0
        score += word_freq[word.lower()]

        if word.lower() in stop_words:
            score -= 100

        if tag.startswith("NNP"):
            score += 1
            score += 2 - word_freq[word.lower()]
            score += sum(1 for char in word if char.isupper())
            if bool(re.search(r"\d", word)):
                score += 1

        if tag.startswith("NN"):
            score += 3
        elif tag.startswith("VB"):
            score += 2
        elif tag.startswith("JJ"):
            score += 1
        elif tag.startswith("CD"):
            score += 3
        elif tag.startswith("DT"):
            score -= 2
        elif tag.startswith("IN"):
            score -= 2
        elif tag.startswith("CC"):
            score -= 2
        word_importance.append((word, score))

    word_important_sorted = sorted(word_importance, key=lambda x: x[1])

    result = " ".join(words)
    while len(result) >= min_length:
        least_important = word_important_sorted.pop(0)
        words.remove(least_important[0])
        result = " ".join(words)

    return result


def split_text_into_logical_sections(text, max_sentences_per_section=5, similarity_threshold=0.3):
    sentences = sent_tokenize(text)

    stop_words = set(stopwords.words("english"))
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        processed_sentences.append(" ".join(words))

    if len(processed_sentences) == 0:
        return [text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    sections = []
    current_section = []

    for i, _processed_sentence in enumerate(processed_sentences):
        current_section.append(sentences[i])

        if len(current_section) >= max_sentences_per_section:
            if i + 1 < len(processed_sentences):
                similarity = similarity_matrix[i, i + 1]
                if similarity < similarity_threshold:
                    sections.append(" ".join(current_section))
                    current_section = []

    if current_section:
        sections.append(" ".join(current_section))

    return sections
