import requests
from bs4 import BeautifulSoup
import fitz
import os
import re
import numpy as np
from pathlib import Path
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from docx import Document
import csv
from openai import OpenAI
import anthropic
import json
from lunr import lunr
import markdown2
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from nltk.corpus import words as nltk_words
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('words')

embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

word_freq = FreqDist(nltk_words.words())

ollama_client = OpenAI( 
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)
openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()

config = {
    "sources": {},
    "perspectives": {},
    "prompts": {},
    'llm': {},
}

browser_request_headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'DNT': '1',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '1',
    'Sec-GPC': '1',
    'Pragma': 'no-cache',
    'Priority': 'u=1',
    'Cache-Control': 'no-cache'
}
browser_session = requests.Session()



def import_configs(dir_path):
    config = {
        "sources": {},
        "perspectives": {},
        "prompts": {},
        'llm': {},
    }
    if not dir_path.endswith('/'):
        dir_path = dir_path + '/'
    for file_name in os.listdir(dir_path):
        file_path = dir_path + file_name
        if file_name == 'perspectives.csv':
            import_perspectives_from_csv(file_path)
        elif file_name == 'llm.txt':
            import_llm_configs_from_txt(file_path)

    import_config_prompts(dir_path+'/prompts')
    import_config_sources(dir_path)

def import_config_prompts(dir_path):
    for file_name in os.listdir(dir_path):
        file_path = dir_path + '/' + file_name
        with open(file_path, 'r') as file:
            prompt_name = file_name.replace('.txt', '')
            prompt = file.read()
            config['prompts'][prompt_name] = prompt

def import_config_sources(dir_path):
    # import sources after prompts and llm stuff
    file_path = dir_path + 'sources.csv'
    if os.path.exists(file_path):
        import_sources_from_csv(file_path)

def import_llm_configs_from_txt(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split(':', 1)
                config['llm'][key.strip()] = value.strip()

def import_sources_from_csv(file_path):
    for row in read_csv_skip_empty(file_path):
        if len(row) >= 3:
            (category, standard, url) = row
            short_standard = standard
            if len(standard) > 100:
                short_standard = shorten_standard_name(standard)
            config['sources'][standard] = ({'category': category, 'standard': short_standard, 'title': standard, 'url': url})
    
def import_perspectives_from_csv(file_path):
    config['perspectives'] = {}
    for row in read_csv_skip_empty(file_path):
        if len(row) >= 2:
            (role, prompt) = row
            config['perspectives'][role] = {'Role': role, 'prompt': prompt}

def read_csv_skip_empty(file_path):
    with open(file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)
        for row in reader:
            cleaned_row = [field.strip().strip('"') for field in row]
            if any(field for field in cleaned_row):
                yield cleaned_row



def make_llm_chat_request(messages,service_name="ollama",model_name="llama3"):
    try:
        if service_name == "openai":
            openai_response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            text_response = openai_response.choices[0].message.content
            if text_response:
                return text_response
            else:
                print(openai_response)
        elif service_name == "ollama":
            ollama_response = ollama_client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            text_response = ollama_response.choices[0].message.content
            if text_response:
                return text_response
            else:
                print(ollama_response)
        elif service_name == "anthropic":
            clean_messages = []
            system_message = ""
            for message in messages:
                if 'role' in message:
                    if message['role'] == 'system':
                        system_message = message['content']
                    else:
                        clean_messages.append(message)
            anthropic_response = anthropic_client.messages.create(
                model=model_name,
                system=system_message,
                messages=clean_messages,
                max_tokens=4096
            )
            text_response = anthropic_response.content[0].text
            if text_response:
                return text_response
            else:
                print(anthropic_response)
        return None
    except Exception as e:
        print(e)
        return None

def find_redirect_in_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
    if meta_refresh:
        redirect_url = re.split(r';\s+url=', meta_refresh['content'])
        if ( len(redirect_url) > 1 ):
            redirect_url = redirect_url[1]
            return redirect_url
    return None

def extract_text_from_url(url,label="",redirect_list=[]):
    if label == "":
        label = url
    if is_pdf(url):
        # print("Extracting PDF: "+url)
        return extract_text_from_pdf(url,label=label)
    elif is_xlsx(url):
        # print("Extracting XLSX: "+url)
        return extract_text_from_xlsx(url,label=label)
    elif is_docx(url):
        # print("Extracting DOCX: "+url)
        return extract_text_from_docx(url,label=label)
    # print("Extracting HTML: "+url)
    return extract_text_from_html(url,label=label,redirect_list=redirect_list)

def extract_text_from_html(url,label="",redirect_list=[]):
    # print("Extracting HTML: "+url)
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    html_file_path = dir_path + fs_safe_url(label) + ".html"
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            text_content = file.read()
            if text_content:
                # print("Found existing text: "+text_file_path)
                return text_content

    content = None
    if os.path.exists(html_file_path):
        with open(html_file_path, 'rb') as file:
            # print("Found existing html: "+html_file_path)
            content = file.read()
    else:
        # print("Querying for new html: "+url)
        try:
            response = browser_session.head(url, headers=browser_request_headers, allow_redirects=True)
            if response.status_code == 200:
                # print("Found new html: "+str(response.status_code)+" "+response.headers.get('content-type'))
                content_type = response.headers.get('content-type')
                if content_type and 'html' in content_type.lower():
                    html_response = browser_session.get(url, headers=browser_request_headers, allow_redirects=True)
                    if html_response.status_code == 200:
                        content = html_response.content
                else:
                    print("Refusing to download unknown content type: "+content_type.lower())
                    return ""
            else:
                print("Failed to download:\n    "+str(response.status_code))
                return ""
        except Exception as e:
            print("Failed to download HTML")
            print(e)
            return ""

    if not content:
        print("Missing content: "+url)
        return ""

    # print("Extracting text from: "+url)
    redirect_url = find_redirect_in_html(content)
    if redirect_url:
        content = ""
        if redirect_url != url and redirect_url.startswith('http') and redirect_url not in redirect_list:
            redirect_list.append(redirect_url)
            return extract_text_from_url(redirect_url,label=label,redirect_list=redirect_list)
        else:
            print("Failed to retrieve redirect: "+url)
            return ""

    with open(html_file_path, 'wb') as file:
        file.write(content)

    soup = BeautifulSoup(content, 'html.parser')
    # Assuming main content is within <main> or <article> tags
    main_content = soup.select_one('#main') or soup.find('main') or soup.find('article') or soup.select_one('#main-content') or soup.select_one('body > div.container') or soup.select_one('body')
    if main_content:
        text_content = main_content.get_text(separator='\n', strip=True)
        with open(text_file_path, 'w') as file:
            file.write(text_content)
        return text_content

    print("Main content not found: "+url)
    return ""

def extract_text_from_pdf(url,label=""):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    pdf_file_path  = dir_path + fs_safe_url(label) + ".pdf"
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            text_content = file.read()
            return text_content

    if not os.path.exists(pdf_file_path):
        if url.startswith('http'):
            # print("Downloading: "+url)
            try:
                response = browser_session.get(url, headers=browser_request_headers, allow_redirects=True)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type')
                    if content_type and 'pdf' in content_type.lower():
                        with open(pdf_file_path, 'wb') as file:
                            file.write(response.content)
                    else:
                        # Check for meta-equiv refresh
                        soup = BeautifulSoup(response.content, 'html.parser')
                        meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
                        if meta_refresh:
                            redirect_url = meta_refresh['content'].split(r';\s*url=')[1]
                            return extract_text_from_pdf(redirect_url, label)
                        else:
                            return "PDF file not found"
                else:
                    return "PDF file not downloaded"
            except Exception as e:
                print("PDF file not downloaded")
                return ""
        else:
            print("PDF file not found")
            return ""

    document = fitz.open(pdf_file_path)
    text_content = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text_content += page.get_text()

    if text_content:
        with open(text_file_path, 'w') as file:
            file.write(text_content)
        return text_content
    else:
        print("PDF content not found")
        return ""

def extract_text_from_xlsx(url,label=""):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    xlsx_file_path = dir_path + fs_safe_url(label) + ".xlsx"
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            text_content = file.read()
            return text_content

    if not os.path.exists(xlsx_file_path):
        if url.startswith('http'):
            # print("Downloading: "+url)
            try:
                response = browser_session.get(url, headers=browser_request_headers, allow_redirects=True)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type')
                    if content_type and 'vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type.lower():
                        with open(xlsx_file_path, 'wb') as file:
                            file.write(response.content)
                    else:
                        # Check for meta-equiv refresh
                        soup = BeautifulSoup(response.content, 'html.parser')
                        meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
                        if meta_refresh:
                            redirect_url = meta_refresh['content'].split(r';\s*url=')[1]
                            return extract_text_from_xlsx(redirect_url, label)
                        else:
                            return "XLSX file not found"
                else:
                    return "XLSX file not downloaded"
            except Exception as e:
                print("XLSX file not downloaded")
                return ""
        else:
            print("XLSX file not found")
            return ""

    text_content = ""
    xls_data = pd.read_excel(xlsx_file_path, engine='openpyxl')
    xls_data.to_csv(text_file_path, index=False)
    # convert_xls_to_csv(xlsx_file_path, text_file_path)

    with open(text_file_path, 'r') as file:
        text_content = file.read()

    if text_content:
        return text_content
    else:
        print("XLSX content not found")
        return ""

def extract_text_from_docx(url,label=""):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    docx_file_path = dir_path + fs_safe_url(label) + ".docx"
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            text_content = file.read()
            return text_content

    if not os.path.exists(docx_file_path):
        if url.startswith('http'):
            # print("Downloading: "+url)
            try:
                response = browser_session.get(url, headers=browser_request_headers, allow_redirects=True)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type')
                    if content_type and 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type.lower():
                        with open(docx_file_path, 'wb') as file:
                            file.write(response.content)
                    else:
                        # Check for meta-equiv refresh
                        soup = BeautifulSoup(response.content, 'html.parser')
                        meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
                        if meta_refresh:
                            redirect_url = meta_refresh['content'].split(r';\s*url=')[1]
                            return extract_text_from_docx(redirect_url, label)
                        else:
                            return "DOCX file not found"
                else:
                    return "DOCX file not downloaded"
            except Exception as e:
                print("DOCX file not downloaded")
                print(e)
                return ""
        else:
            print("DOCX file not found")
            return ""

    text_content = ""

    doc = Document(docx_file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    text_content = '\n'.join(text)

    if text_content:
        with open(text_file_path, 'w') as file:
            file.write(text_content)
        return text_content
    else:
        print("DOCX content not found")
        return ""




def fs_safe_url(url):
    return url.replace('/', '_').replace(':', '_')

def is_pdf(url):
    url_safe = url.split('?')[0].split('#')[0]
    if url_safe.endswith('.pdf'):
        return True
    response = browser_session.head(url, headers=browser_request_headers, allow_redirects=True)
    content_type = response.headers.get('content-type')
    if content_type and 'pdf' in content_type.lower():
        return True
    return False

def is_xlsx(url):
    url_safe = url.split('?')[0].split('#')[0]
    if url_safe.endswith('.xlsx'):
        return True
    response = browser_session.head(url, headers=browser_request_headers, allow_redirects=True)
    content_type = response.headers.get('content-type')
    if content_type and 'vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type.lower():
        return True
    return False

def is_docx(url):
    url_safe = url.split('?')[0].split('#')[0]
    if url_safe.endswith('.docx'):
        return True
    response = browser_session.head(url, headers=browser_request_headers, allow_redirects=True)
    content_type = response.headers.get('content-type')
    if content_type and 'vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type.lower():
        return True
    return False



def shorten_standard_name(standard):
    return shorten_standard_name_via_nltk(standard)

def shorten_standard_name_via_llm(standard):
    user_prompt = config['prompts']['shorten_standard_name']
    user_prompt += standard
    response = make_llm_chat_request(
        service_name=config['llm']['chat_service_name'],
        model_name=config['llm']['chat_model_name'],
        messages=[
            {"role": "user", "content": user_prompt},
        ]
    )
    if response:
        return response
    return ""

def shorten_standard_name_via_nltk(standard, min_length=200):
    # Tokenize the sentence
    words = word_tokenize(standard)
    
    # Get part-of-speech tags
    tagged_words = pos_tag(words)
    
    # Define stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(['federal', 'government', 'office', 'agency', 'Memorandum', 'requirement', 'requirements', 'policy', 'policies'])

    # Create a list of word importances (lower score is less important)
    word_importance = []
    for word, tag in tagged_words:
        score = 0
        score += word_freq[word.lower()]
        
        if word.lower() in stop_words:
            score -= 100

        if tag.startswith('NNP'):  # Nouns
            # one for being nnp
            score += 1
            # 0-2 based on commonality
            score += ( 2 - word_freq[word.lower()] )
            # one for each capitalization
            score += sum(1 for char in word if char.isupper())
            # one for having digits
            if bool(re.search(r'\d', word)):
                score += 1
        
        if tag.startswith('NN'):  # nouns
            score += 3
        elif tag.startswith('VB'):  # Verbs
            score += 2
        elif tag.startswith('JJ'):  # Adjectives
            score += 1
        elif tag.startswith('CD'):  # Digit ?
            score += 3
        elif tag.startswith('DT'):  # the
            score -= 2
        elif tag.startswith('IN'):  # of
            score -= 2
        elif tag.startswith('CC'):  # and
            score -= 2
        word_importance.append((word, score))

    word_important_sorted = sorted(word_importance, key=lambda x: x[1])

    result = ' '.join(words)
    while len(result) >= min_length:
        # Find the word with the lowest importance score
        least_important = word_important_sorted.pop(0)
        # Remove the least important word
        # print( "removing "+ least_important[0] )
        words.remove(least_important[0])
        result = ' '.join(words)
    
    return result




def split_text_into_overlapping_chunks(text, max_length, overlap=0):
    words = text.split()
    chunks = []
    step = max_length - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + max_length]
        chunks.append(' '.join(chunk))
        if len(chunk) < max_length:
            break
    return chunks

def split_text_into_logical_sections(text, max_sentences_per_section=5, similarity_threshold=0.3):
    sentences = sent_tokenize(text)
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        processed_sentences.append(' '.join(words))
    
    if len(processed_sentences) == 0:
        return [text]

    # Calculate TF-IDF vectors for sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    
    # Calculate cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    sections = []
    current_section = []
    
    for i, processed_sentence in enumerate(processed_sentences):
        current_section.append(sentences[i])
        
        if len(current_section) >= max_sentences_per_section:
            # Check similarity with the next sentence
            if i + 1 < len(processed_sentences):
                similarity = similarity_matrix[i, i+1]
                if similarity < similarity_threshold:
                    sections.append(' '.join(current_section))
                    current_section = []
    
    # Add any remaining sentences to the last section
    if current_section:
        sections.append(' '.join(current_section))
    
    return sections



def generate_main_embeddings():
    print("Generating embedding for everything")
    main_embedding_file_path = "./assets/embedding.json"
    main_embeddings = []
    for (standard,source) in config['sources'].items():
        url = source['url']
        label = source['standard']
        if not url:
            continue
        
        dir_path = "./sources/" + fs_safe_url(label) + "/"
        text_file_path = dir_path + fs_safe_url(label) + ".txt"
        embedding_file_path = dir_path + "embedding.json"

        standard_embeddings = []
        if os.path.exists(embedding_file_path):
            with open(embedding_file_path, 'r') as file:
                file_embeddings = file.read()
                json_embeddings = json.loads(file_embeddings)
                if json_embeddings:
                    standard_embeddings = json_embeddings

        overall_summary = ""
        prompt_name = "overall"
        summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.txt")
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as file:
                overall_summary = file.read()


        keyword_summary = ""
        prompt_name = "keywords"
        summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.txt")
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as file:
                keyword_summary = file.read()

        safe_label = fs_safe_url(label)

        if not standard_embeddings and not keyword_summary and not overall_summary:
            continue
        
        overall_embedding = np.zeros(512)
        for section in standard_embeddings:
            embedding = section['embedding'][0]
            if isinstance(embedding, (list, np.ndarray)) and len(embedding) == 512:
                overall_embedding += np.array(embedding)
        # add in embeddings from keywords ?
        # add in embeddings from summary  ?
        overall_embedding = overall_embedding.tolist()

        main_embeddings.append({
            "id": safe_label,
            "title": label,
            "body": overall_summary,
            "keywords": keyword_summary,
            "embedding": overall_embedding
        })

    with open(main_embedding_file_path, 'w') as f:
        json.dump(main_embeddings, f)

def generate_embeddings_for_url(url,label=""):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    embed_file_path = dir_path + "/embedding.json"

    if os.path.exists(embed_file_path):
        return

    text = extract_text_from_url(url,label=label)
    if not text:
        return

    embedding = generate_embeddings_for_text_sections(text)
    with open(embed_file_path, 'w') as f:
        json.dump(embedding, f)

def generate_embeddings_for_text_sections(text):
    return generate_embeddings_for_text_sections_via_use(text)
    # return generate_embeddings_for_text_sections_via_bert(text)

def generate_embeddings_for_text_sections_via_use(text):
    chunks = split_text_into_logical_sections(text, max_sentences_per_section=10, similarity_threshold=0.3)
    embeddings = []
    for chunk_id,chunk in enumerate(chunks):
        chunk_embedding = embed_model_use([chunk])
        chunk_embedding_list = chunk_embedding.numpy().tolist()
        embeddings.append({
            'id': chunk_id,
            'text': chunk, 
            'embedding': chunk_embedding_list
        })
    return embeddings

def generate_embeddings_for_text_sections_via_bert(text):
    chunks = split_text_into_logical_sections(text, max_sentences_per_section=10, similarity_threshold=0.4)
    embeddings = []
    for i,chunk in enumerate(chunks):
        inputs = embed_tokenizer_bert(chunk, max_length=500, return_tensors='pt', truncation=True, padding='max_length',)
        with torch.no_grad():
            outputs = embed_model_bert(**inputs)
        embeddings.append({
            'id': i,
            'text': chunk,
            'embedding':outputs.last_hidden_state[:, 0, :].numpy().tolist()[0]
        })
    return embeddings


def generate_summaries_for_url(url,label=""):
    document_text = extract_text_from_url(url,label=label)
    if not document_text:
        return

    # Generate summaries from text source
    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    # inject the text source into the prompt
    system_prompt = config['prompts']['system_context'].format(document_text=document_text)

    perspectives = config['perspectives']
    summaries = {
        "overall": "",
        "punchline": "",
        "keywords": "",
        "actions": {}
    }
    
    # Generate one overall summary
    summaries['overall'] = generate_overall_summary(system_prompt, text_file_path)

    # Generate one single punchline summary, including results for each perspective
    summaries['punchline'] = generate_punchline_summary(system_prompt, text_file_path, perspectives)

    # Generate multiple action summaries, one from each perspective
    summaries['actions'] = generate_action_summaries(system_prompt, text_file_path, perspectives)

    summaries['keywords'] = generate_keyword_summary(document_text, system_prompt, text_file_path)

    return summaries

def generate_overall_summary(system_prompt, text_file_path):
    # Generate one overall summary
    prompt_name = "overall"
    summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.txt")
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'r') as file:
            return file.read()
    print("Generating overall summary")
    user_prompt = config['prompts'][prompt_name]
    response = make_llm_chat_request(
        service_name=config['llm']['chat_service_name'],
        model_name=config['llm']['chat_model_name'],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if response:
        with open(summary_file_path, 'w') as file:
            file.write(response)
        return response
    else:
        print("Failed to generate summary "+summary_file_path)
        return ""

def generate_punchline_summary(system_prompt, text_file_path, perspectives):
    # Generate one single punchline summary, but including results for each perspective
    prompt_name = "punchline"
    summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.txt")
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'r') as file:
            return file.read()
    user_prompt = config['prompts'][prompt_name] + "\n- "
    user_prompt += "\n- ".join(perspective_data['prompt'] for perspective_data in perspectives.values())
    print("Generating punchline summaries")
    response = make_llm_chat_request(
        service_name=config['llm']['chat_service_name'],
        model_name=config['llm']['chat_model_name'],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if response:
        with open(summary_file_path, 'w') as file:
            file.write(response)
        return response
    else:
        print("Failed to generate summary "+summary_file_path)
        return ""

def generate_action_summaries(system_prompt, text_file_path, perspectives):
    # Generate multiple action summaries, one from each perspective
    results = {}
    for perspective,perspective_data in perspectives.items():
        results[perspective] = generate_action_summary(system_prompt, text_file_path, perspective, perspective_data['prompt'])
    return results

def generate_action_summary(system_prompt, text_file_path, perspective, perspective_prompt):
    # Generate action summary, from a single perspective
    prompt_name = "actions"
    summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.{perspective}.txt")
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'r') as file:
            return file.read()
    user_prompt = config['prompts'][prompt_name]
    user_prompt += "\n Consider things from only this one perspective:"
    user_prompt += "\n"+ perspective_prompt
    print("Generating action summary: "+perspective)
    response = make_llm_chat_request(
        service_name=config['llm']['chat_service_name'],
        model_name=config['llm']['chat_model_name'],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if response:
        with open(summary_file_path, 'w') as file:
            file.write(response)
        return response
    else:
        print("Failed to generate summary "+summary_file_path)
        return ""

def generate_keyword_summary(document_text, system_prompt, text_file_path):
    generate_keyword_summary_via_llm(system_prompt, text_file_path)
    # generate_keyword_summary_via_bert(document_text, text_file_path)
    # generate_keyword_summary_via_spacy(document_text, text_file_path)
    pass

def generate_keyword_summary_via_llm(system_prompt, text_file_path):
    # Generate action summary, from a single perspective
    prompt_name = "keywords"
    summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.txt")
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'r') as file:
            return file.read()
    user_prompt = config['prompts'][prompt_name]
    print("Generating keyword summary")
    response = make_llm_chat_request(
        service_name=config['llm']['chat_service_name'],
        model_name=config['llm']['chat_model_name'],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if response:
        with open(summary_file_path, 'w') as file:
            file.write(response)
        return response
    else:
        print("Failed to generate summary "+summary_file_path)
        return ""


def generate_search_index():
    generate_lunr_index()

def generate_lunr_index():
    print("Generating search index for everything")
    search_documents = []
    for (standard,source) in config['sources'].items():
        url = source['url']
        label = source['standard']
        if not url:
            continue
        
        dir_path = "./sources/" + fs_safe_url(label) + "/"
        text_file_path = dir_path + fs_safe_url(label) + ".txt"

        overall_summary = ""
        prompt_name = "overall"
        summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.txt")
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as file:
                overall_summary = file.read()

        keyword_summary = ""
        prompt_name = "keywords"
        summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.txt")
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as file:
                keyword_summary = file.read()

        safe_label = fs_safe_url(label)

        if not overall_summary and not keyword_summary:
            continue

        # searchable_content = overall_summary + " " + keyword_summary

        search_documents.append({
            "id": safe_label,
            "title": label,
            "body": overall_summary,
            "keywords": keyword_summary
        })

    index = lunr(
        ref='id',
        fields=['title', 'body', 'keywords'],
        documents=search_documents
    )
    index_data = index.serialize()
    with open('./assets/lunr_index.json', 'w') as file:
        json.dump(index_data, file)


def generate_index_page_for_url(url,label=""):
    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    index_file_path = dir_path + "/index.html"

    text_file_url = './' + fs_safe_url(label) + ".txt"
    text_file_path = dir_path + '/' + fs_safe_url(label) + ".txt"

    text = ""
    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            text = file.read()
    chunks = split_text_into_logical_sections(text, max_sentences_per_section=10, similarity_threshold=0.3)
    text_chunks = ""
    for chunk_id,chunk in enumerate(chunks):
        text_chunks += f"""<div id="chunk-{chunk_id}" class="text-chunk" /><a name="chunk-{chunk_id}"><sup>[{chunk_id}]</sup></a> {chunk}</div>"""

    summaries_html = ""

    # gather the prompt names and prompts for display
    prompts = {
        "overall": config['prompts']['overall'],
        "punchline": config['prompts']['punchline'] + "\n- " + "\n- ".join(perspective_data['prompt'] for perspective_data in config['perspectives'].values())
    }
    for perspective,perspective_data in config['perspectives'].items():
        prompt_name = "actions." + perspective
        user_prompt = config['prompts']['actions']
        user_prompt += "\n Consider things from only this one perspective:"
        user_prompt += "\n"+ perspective_data['prompt']
        prompts[prompt_name] = user_prompt

    # render each prompt and it's response
    for prompt_name,prompt in prompts.items():
        summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.txt")
        if os.path.exists(summary_file_path):
            summary_file_text = ""
            with open(summary_file_path, 'r') as file:
                summary_file_text = file.read()
            summary_html = markdown2.markdown(summary_file_text)
            # prompt = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', prompt)
            # prompt = markdown2.markdown(prompt)
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

    menu_html = f"""
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

        <script src="../../assets/tf.js" type="text/javascript"></script>
        <script src="../../assets/tf-universal-sentence-encoder.js" type="text/javascript"></script>
        <script src="../../assets/page_embedding_search.js" type="text/javascript"></script>
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
                    <div id="embed-query">
                        <input type="text" id="embed-query-input"/>
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

    with open(index_file_path, 'w') as file:
        file.write(index_tmpl)

def generate_main_index_page():
    index_file_path = "./index.html"

    # gather the prompt names and prompts for display
    prompts = {
        "overall": config['prompts']['overall'],
        "punchline": config['prompts']['punchline'],
        "keywords": config['prompts']['keywords'],
    }
    for perspective,perspective_data in config['perspectives'].items():
        prompt_name = "actions." + perspective
        user_prompt = config['prompts']['actions']
        user_prompt += "\n Consider things from only this one perspective:"
        user_prompt += "\n"+ perspective_data['prompt']
        prompts[prompt_name] = user_prompt
        prompts["punchline"] += "\n- " + perspective

    # render each prompt and it's response
    prompts_html = ""
    for prompt_name,prompt in prompts.items():
        prompt = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', prompt)
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
    for (standard,source) in config['sources'].items():
        url = str(source['url'])
        if not url:
            continue
        standard_index_file_path = "./sources/" + fs_safe_url(standard) + "/index.html"
        sources_js[standard] = standard_index_file_path

    sources_js = json.dumps(sources_js)
    with open("./assets/sources.js", "w") as file:
        file.write(f"var sources = {sources_js};")

    page_sources_js = {}
    for (standard,source) in config['sources'].items():
        url = str(source['url'])
        if not url:
            continue
        standard_index_file_path = "../" + fs_safe_url(standard) + "/index.html"
        page_sources_js[standard] = standard_index_file_path

    page_sources_js = json.dumps(page_sources_js)
    with open("./assets/page_sources.js", "w") as file:
        file.write(f"var sources = {page_sources_js};")


    menu_html = f"""
        <div id="nav-menu" class="accordion" role="navigation" aria-label="Page Navigation">
            <div class="accordion-item">
                <button id="nav-menu-toggle" class="accordion-header"><span class="accordion-header-text">Standards</span><span class="accordion-header-icon"></span></button>
                <div class="accordion-content"><ul id="nav-menu-standards"></ul></div>
            </div>
        </div>
        <br /><br />
    """


    # render the index page
    index_tmpl = f"""<html>
        <link rel="stylesheet" type="text/css" href="./assets/standards.css" />
        <script src="./assets/standards.js"></script>

        <script src="./assets/sources.js"></script>
        <script src="./assets/nav.js"></script>

        <script src="./assets/tf.js" type="text/javascript"></script>
        <script src="./assets/tf-universal-sentence-encoder.js" type="text/javascript"></script>
        <script src="./assets/embedding_search.js"></script>
    <body>
        <h1>Gov Doc Summaries</h1>
        
        {prompts_html}

        {menu_html}

    </body>
    </html>"""

    with open(index_file_path, 'w') as file:
        file.write(index_tmpl)




def process_sources():
    for (standard,source) in config['sources'].items():
        # if standard != "The HTTPS-Only Standard":
        #     continue
        url = source['url']
        std = source['standard']
        if not url:
            continue
        print("Processing: "+std)
        extract_text_from_url(url,label=std)
        generate_embeddings_for_url(url,label=std)
        generate_summaries_for_url(url,label=std)
        generate_index_page_for_url(url,label=std)
    # generate_search_index()
    generate_main_embeddings()
    generate_main_index_page()


import_configs('./config')


embed_model_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# embed_model = await tf.loadGraphModel('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', {fromTFHub: true});
# embed_tokenizer = await (await fetch('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')).json();
            
embed_model_bert = BertModel.from_pretrained(config['llm']['embed_model_name'])
embed_tokenizer_bert = BertTokenizer.from_pretrained(config['llm']['embed_model_name'])

keyword_model = BertModel.from_pretrained(config['llm']['keyword_model_name'])
keyword_tokenizer = BertTokenizer.from_pretrained(config['llm']['keyword_model_name'])

nlp = spacy.load("en_core_web_sm")

process_sources()
