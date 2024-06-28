import requests
from bs4 import BeautifulSoup
import fitz
import os
import re
from pathlib import Path
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from docx import Document
import csv
from openai import OpenAI
import anthropic
import json
import markdown2

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
    main_content = soup.select_one('#main') or soup.find('main') or soup.find('article') or soup.select_one('#main-content') or soup.select_one('body > div.container')or soup.select_one('body')
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



def generate_embeddings_for_url(url,label=""):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    text_file_path = dir_path + fs_safe_url(label) + ".txt"
    embed_file_path = text_file_path.replace(".txt", ".embed")

    if os.path.exists(embed_file_path):
        return

    text = extract_text_from_url(url,label=label)
    if not text:
        return

    final_embedding = generate_embeddings_for_text(text)
    torch.save(final_embedding, embed_file_path)

def generate_embeddings_for_text(text):
    max_tokens = 500
    token_overlap_window = 250
    chunks = split_text_into_overlapping_chunks(text, max_tokens, overlap=token_overlap_window)
    embeddings = []
    for chunk in chunks:
        inputs = embed_tokenizer(chunk, return_tensors='pt', truncation=True, padding='max_length', max_length=max_tokens)
        with torch.no_grad():
            outputs = embed_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return torch.cat(embeddings, dim=1)

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
        "actions": {}
    }
    
    # Generate one overall summary
    summaries['overall'] = generate_overall_summary(system_prompt, text_file_path)

    # Generate one single punchline summary, including results for each perspective
    summaries['punchline'] = generate_punchline_summary(system_prompt, text_file_path, perspectives)

    # Generate multiple action summaries, one from each perspective
    summaries['actions'] = generate_action_summaries(system_prompt, text_file_path, perspectives)
    
    return summaries

def generate_overall_summary(system_prompt, text_file_path):
    # Generate one overall summary
    prompt_name = "overall"
    summary_file_path = text_file_path.replace(".txt", f".{config['llm']['chat_model_name']}.summary.{prompt_name}.txt")
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'r') as file:
            return file.read()
    else:        
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
    else:
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
    else:
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



def generate_index_page(url,label=""):
    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    index_file_path = dir_path + "/index.html"

    text_file_url = './' + fs_safe_url(label) + ".txt"
    text_file_path = dir_path + '/' + fs_safe_url(label) + ".txt"

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
                # <div class="accordion">
                #     <div class="accordion-item">
                #         <button class="accordion-header">{summary_title} Prompt</button>
                #         <div class="accordion-content">{prompt_html}</div>
                #     </div>
                # </div>

    # render the index page


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
        <link rel="stylesheet" type="text/css" href="../../assets/standard.css" />
        <script src="../../assets/standard.js" type="text/javascript"></script>
        <script src="../../assets/sources.js" type="text/javascript"></script>
        <script src="../../assets/nav.js" type="text/javascript"></script>
    <body>
        <h1>{label}</h1>

        {menu_html}

        <div class="accordion">
            <div class="accordion-item">
                <button class="accordion-header">Source Data</button>
                <div class="accordion-content">
                    <br />
                    <a href="{url}">Raw Data</a>
                    <br /><br />
                    <a href="{text_file_url}">Source Text</a>
                    <br /><br />
                </div>
            </div>
        </div>
        <br /><br />

        {summaries_html}

    </body>
    </html>"""

    with open(index_file_path, 'w') as file:
        file.write(index_tmpl)

def generate_index_index_page():
    index_file_path = "./index.html"

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
        url = source['url']
        if not url:
            continue
        standard_index_file_path = "./sources/" + fs_safe_url(standard) + "/index.html"
        sources_js[standard] = standard_index_file_path

    sources_js = json.dumps(sources_js)
    with open("./assets/sources.js", "w") as file:
        file.write(f"var sources = {sources_js};")

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
        <link rel="stylesheet" type="text/css" href="./assets/standard.css" />
        <script src="./assets/standard.js"></script>
        <script src="./assets/sources.js"></script>
        <script src="./assets/nav.js"></script>
    <body>
        <h1>Gov Doc Summaries</h1>

        
        {prompts_html}

        {menu_html}

    </body>
    </html>"""

    with open(index_file_path, 'w') as file:
        file.write(index_tmpl)


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

def shorten_standard_name(standard):
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

def import_sources_from_csv(file_path):
    for row in read_csv_skip_empty(file_path):
        if len(row) >= 3:
            (category, standard, url) = row
            short_standard = standard
            if len(standard) > 100:
                short_standard = shorten_standard_name(standard)
            config['sources'][standard] = ({'category': category, 'standard': short_standard, 'title': standard, 'url': url})
    
def import_perspectives_from_csv(file_path):
    perspectives = {}
    for row in read_csv_skip_empty(file_path):
        if len(row) >= 2:
            (role, prompt) = row
            perspectives[role] = {'Role': role, 'prompt': prompt}
    return perspectives

def read_csv_skip_empty(file_path):
    with open(file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)
        for row in reader:
            cleaned_row = [field.strip().strip('"') for field in row]
            if any(field for field in cleaned_row):
                yield cleaned_row



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
        # generate_embeddings_for_url(url,label=std)
        generate_summaries_for_url(url,label=std)
        generate_index_page(url,label=std)
    generate_index_index_page()




import_configs('./config')

embed_model = BertModel.from_pretrained(config['llm']['embed_model_name'])
embed_tokenizer = BertTokenizer.from_pretrained(config['llm']['embed_model_name'])

print(config.keys())

# process_sources()
