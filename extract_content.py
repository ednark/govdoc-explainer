import requests
from bs4 import BeautifulSoup
import fitz
import os
import re
from pathlib import Path
import urllib
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from docx import Document
from openai import OpenAI

openai_client = OpenAI( 
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)
# openai_client = OpenAI()

embed_model_name = "bert-base-uncased"
embed_model = BertModel.from_pretrained(embed_model_name)
embed_tokenizer = BertTokenizer.from_pretrained(embed_model_name)

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
    text = extract_text_from_url(url,label=label)
    if not text:
        return
    # Generate summary

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    system_prompt = f"""
        Context: I am a tech writer for a small application development company. My company is a government contractor who creates and hosts web applications with externally facing public websites.
        Objective: Ensure that our team understands relevant ideas and requirements from the provided Government Standards Document.
        Audience: Non-technical business stakeholders and writers who need to check that proposals are adhering relevant portions of the government standards.
        Government Standards Document: {text}
    """
    common_user_prompts = {
        "summary": """
            Using the included standards document, produce a document summary in simple direct and unambiguous language:
            •	List only the primary standards, guidelines, and requirements mentioned in the document.
            •	Highlight sections that focus on security, accessibility, user experience, and compliance.
            •	Summarize each part of the result into an even shorter result.
        """
    }
    perspective_user_prompts = {
        "actions": """
            In practical terms for a web application, what minimal list of actions must I take on a project to show compliance with standards? Each item should itself be a short summary. 
        """
    }
    collective_user_prompts = {
        "punchline": """
            In a short sentence that can be used as an elevtor pitch, what is the purpose of this document and what is it trying to achieve? Please describe in simple plain language that a non-technical business person could understand. Simplify the purpose in single sentence leaving out introductory information like the title of the standard.
        """
    }
    perspectives = {
        "DevOps": "a DevOps technician, who must design and manage infrastructure to support the application.",
        "Designer": "a Designer, who specializes in Human Centered Design and will be responsible for the User Experience and User Interface of the application.",
        "Developer": "a Developer, who specializes in implementing Drupal, who will have to implement the backend business logic speciifed for an application.",
        "Project Manager": "a Project manager who will need to track and organize work and guide the client towards completing a task."
    }
    results = {
        "punchline": "",
        "actions": "",
        "summary": ""
    }
    
    # common non-perspective summaries
    for prompt_name,user_prompt in common_user_prompts.items():
        # Generate one overall summary
        summary_file_path = text_file_path.replace(".txt", f".summary-{prompt_name}.summary")
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as file:
                summary = file.read()
                results[prompt_name] = summary
                continue
        print("Generating common summary: "+prompt_name)
        openai_response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        summary = openai_response.choices[0].message.content
        if summary:
            results[prompt_name] = summary
            with open(summary_file_path, 'w') as file:
                file.write(summary)
        else:
            print("Failed to generate summary "+summary_file_path)
            print(openai_response)

    # collective summaries
    for prompt_name,user_prompt in collective_user_prompts.items():
        summary_file_path = text_file_path.replace(".txt", f".summary-{prompt_name}.summary")
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as file:
                summary = file.read()
                results[prompt_name] = summary
                continue
        user_prompt = "Generate a summary or each of these perspectives: "+ ". ".join(perspectives.values()) +"Include only inormation relevant to each perspective, ignoring and leaving out any details not important from that  perspective."
        print("Generating collective summary: "+prompt_name)
        openai_response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        summary = openai_response.choices[0].message.content
        if summary:
            results[prompt_name] = summary
            with open(summary_file_path, 'w') as file:
                file.write(summary)
        else:
            print("Failed to generate summary "+summary_file_path)
            print(openai_response)
    
    # individual per-perspective summaries
    for prompt_name,user_prompt in perspective_user_prompts.items():
        for perspective,perspective_prompt in perspectives.items():
            summary_file_path = text_file_path.replace(".txt", f".summary-{prompt_name}-{perspective}.summary")
            user_perspective_prompt = f"""From the perspective of {perspective_prompt} Ignore and leave out any information not important from this perspective. {user_prompt}"""
            if os.path.exists(summary_file_path):
                with open(summary_file_path, 'r') as file:
                    summary = file.read()
                    results[prompt_name] = summary
                    continue
            print("Generating "+ perspective+" summary: "+prompt_name)
            openai_response = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_perspective_prompt},
                ]
            )
            summary = openai_response.choices[0].message.content
            if summary:
                results[prompt_name] = summary
                with open(summary_file_path, 'w') as file:
                    file.write(summary)
            else:
                print("Failed to generate summary "+summary_file_path)
                print(openai_response)

    return results

def generate_index_page(url,label=""):
    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    index_file_path = dir_path + "/index.html"

    text_file_path = './' + fs_safe_url(label) + ".txt"

    common_user_prompts = ["summary"]
    perspective_user_prompts = ["actions"]
    collective_user_prompts = ["punchline"]
    perspectives = {
        "DevOps": "a DevOps technician, who must design and manage infrastructure to support the application.",
        "Designer": "a Designer, who specializes in Human Centered Design and will be responsible for the User Experience and User Interface of the application.",
        "Developer": "a Developer, who specializes in implementing Drupal, who will have to implement the backend business logic speciifed for an application.",
        "Project Manager": "a Project manager who will need to track and organize work and guide the client towards completing a task."
    }

    summaries = ""

    # common non-perspective summaries
    summaries += f"""
        <h3>Common</h3>
    """
    for prompt_name in common_user_prompts:
        # Generate one overall summary
        summary_file_path = text_file_path.replace(".txt", f".summary-{prompt_name}.summary")
        summaries += f"""
            <a href="{summary_file_path}">{prompt_name}</a><br />
        """

    # collective summaries
    summaries += f"""
        <h3>Collective</h3>
    """
    for prompt_name in collective_user_prompts:
        summary_file_path = text_file_path.replace(".txt", f".summary-{prompt_name}.summary")
        summaries += f"""
            <a href="{summary_file_path}">{prompt_name}</a><br />
        """

    # individual per-perspective summaries
    summaries += f"""
        <h3>Perspectives</h3>
    """
    for prompt_name in perspective_user_prompts:
        for perspective,perspective_prompt in perspectives.items():
            summary_file_path = text_file_path.replace(".txt", f".summary-{prompt_name}-{perspective}.summary")
            summaries += f"""
            <a href="{summary_file_path}">{prompt_name} - {perspective}</a><p>{perspective_prompt}</p><br />
            """

    index_tmpl = f"""<html>
        <link rel="stylesheet" type="text/css" href="../../assets/standard.css" />
        <script src="../../assets/standard.js"></script>
    <body>
        <h1>{label}</h1>

        <h2>Source</h3><a href="{url}">Raw Data</a>
        <h2>Source Text</h3><a href="{text_file_path}">Text Only</a>

        {summaries}

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


def read_list_of_sources():
    sources = {}
    with open('sources.txt', 'r') as file:
        lines = file.readlines()
    category = ''
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('Category: '):
            category = line.replace('Category: ', '')
            continue
        
        parts = line.split('\t', 1)
        if len(parts) == 2:
            (standard, url) = parts
            sources[standard] = ({'category': category, 'standard': standard, 'url': url})
    return sources

def process_sources():
    sources = read_list_of_sources()
    for (standard,source) in sources.items():
        if standard != "The HTTPS-Only Standard":
            continue
        url = source['url']
        std = source['standard']
        if not url:
            continue
        print("Processing: "+std)
        extract_text_from_url(url,label=std)
        # generate_embeddings_for_url(url,label=std)
        # generate_summaries_for_url(url,label=std)
        generate_index_page(url,label=std)

process_sources()

# url = 'https://www.whitehouse.gov/wp-content/uploads/legacy_drupal_files/omb/memoranda/2017/m-17-06.pdf'
# text = extract_text_from_url(url)
# # print(text)

# url = 'https://obamawhitehouse.archives.gov/sites/default/files/omb/egov/digital-government/digital-government.html'
# content = extract_text_from_url(url)
# # print(content)

## Possible prompts:

# I am a developer and tech writer for a small application development company. The company is a government contractor that creates and hosts web appliations with externally facing public websites for the U.S. government.
# I need to make sure our project proposals incorporate ideas from the standards doc.
# What portions of the document are most relevant to my company's proposal?
# Which parts of the standard are most relevant to addess in order to make sure a web-based application is compliant?

# In practical terms, what types of thing must I do to show my project is complying with those parts of the standard?

# What parts of this standard are unique to this document, and not addressed in other government standards documents?

# In a short paragraph, what is the purpose of this document and what is it trying to achieve?
# Please describe in plain language that a non-technical business person could understand.


# Identify Key Requirements:
# 	•	List the primary standards, guidelines, and requirements mentioned in the document.
# 	•	Highlight sections that focus on security, accessibility, user experience, and compliance, as these are often critical for government projects.


## long prompt better ???

# Prompt:
#   Identifying Relevant Portions of the Standards Document for Project Proposals
# Context:
#   You are a developer and tech writer for a small application development contractor that creates applications with externally facing public websites for the U.S. government. You need to ensure that your project proposals incorporate ideas from a specific standards document.
# Objectives:
# 	1.	Understand the key requirements and guidelines outlined in the standards document.
# 	2.	Identify portions of the document that are most relevant to the specific needs and objectives of your company’s project proposals.
# 	3.	Ensure that the proposals align with the standards to increase the likelihood of acceptance and compliance.
# Steps:
# 	1.	Document Overview:
# 	•	Begin by reading the introduction and table of contents of the standards document.
# 	•	Summarize the overall purpose and scope of the document in a few sentences.
# 	•	Note any sections or chapters that seem particularly relevant to your company’s work in developing public-facing applications.
# 	2.	Identify Key Requirements:
# 	•	List the primary standards, guidelines, and requirements mentioned in the document.
# 	•	Highlight sections that focus on security, accessibility, user experience, and compliance, as these are often critical for government projects.
# 	3.	Align with Company Objectives:
# 	•	Review your company’s typical project objectives and requirements.
# 	•	Compare these objectives with the standards document to identify overlapping areas.
# 	•	Mark sections that directly relate to your company’s strengths, such as specific technologies, methodologies, or compliance practices your company excels in.
# 	4.	Detailed Analysis:
# 	•	For each relevant section identified, summarize the key points and how they apply to your proposals.
# 	•	Provide specific examples of how your company’s past projects have adhered to these standards or how future proposals can be adapted to meet them.
# 	•	Note any potential gaps or areas where your company may need to improve or adjust its approach to fully comply with the standards.
# 	5.	Implementation Plan:
# 	•	Create a plan for incorporating the identified standards into your project proposals.
# 	•	Outline steps for ensuring ongoing compliance, including regular reviews and updates to proposals as standards evolve.
# 	•	Suggest any additional training or resources your team might need to stay current with the standards.
# Deliverable:
#   A comprehensive report or guide that details the relevant portions of the standards document, how they align with your company’s objectives, and actionable steps for incorporating these standards into your project proposals.
# Example:
# Standards Document Analysis for [Your Company Name]
# 1. Document Overview:
# 	•	Purpose: To ensure all public-facing applications meet federal standards for security, accessibility, and user experience.
# 	•	Relevant Sections: Security (Chapter 3), Accessibility (Chapter 5), User Experience (Chapter 7)
# 2. Key Requirements:
# 	•	Security: Must comply with NIST guidelines.
# 	•	Accessibility: Must meet WCAG 2.1 AA standards.
# 	•	User Experience: Must follow best practices for usability and user-centered design.
# 3. Alignment with Company Objectives:
# 	•	Security: Our company uses industry-standard encryption and follows NIST guidelines.
# 	•	Accessibility: Our applications are designed to be fully accessible, with a history of compliance with WCAG standards.
# 	•	User Experience: We prioritize user-centered design in all projects.
# 4. Detailed Analysis:
# 	•	Security (Chapter 3): Implement multi-factor authentication, regular security audits.
# 	•	Accessibility (Chapter 5): Use ARIA landmarks, ensure keyboard navigability.
# 	•	User Experience (Chapter 7): Conduct user testing, ensure intuitive navigation.
# 5. Implementation Plan:
# 	•	Regularly review proposals against standards.
# 	•	Train team on updates to NIST and WCAG guidelines.
# 	•	Schedule quarterly compliance audits.


# pip install python-docx
# application/vnd.openxmlformats-officedocument.wordprocessingml.document
# from docx import Document
# def extract_text_from_docx(file_path):
#     doc = Document(file_path)
#     text = []
#     for paragraph in doc.paragraphs:
#         text.append(paragraph.text)
#     return '\n'.join(text)
# file_path = 'path/to/your/document.docx'
# text = extract_text_from_docx(file_path)
# print(text)

# pip install pandas xlrd
# import pandas as pd

# def convert_xls_to_csv(xlsx_file_path, csv_file_path):
#     # Read the .xls file
#     xls_data = pd.read_excel(xlsx_file_path, engine='xlrd')
#     # Write the data to a .csv file
#     xls_data.to_csv(csv_file_path, index=False)
# # Specify the path to your .xls file and the desired .csv file
# xlsx_file_path = 'path/to/your/file.xls'
# csv_file_path = 'path/to/your/file.csv'
# # Convert the file
# convert_xls_to_csv(xlsx_file_path, csv_file_path)
# print(f'Converted {xlsx_file_path} to {csv_file_path}')