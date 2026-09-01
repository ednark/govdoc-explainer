import os
import re
import shutil
from pathlib import Path

import pandas as pd
import pymupdf
import requests
import urllib3
from bs4 import BeautifulSoup
from docx import Document

from govdoc_explainer.text_utils import fs_safe_url

# Some .gov hosts (e.g. fam.state.gov) serve misconfigured certificate chains.
# This tool only fetches public documents at build time, so we skip cert
# verification rather than fail the whole ingestion on a bad chain.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

browser_request_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-GPC": "1",
    "Pragma": "no-cache",
    "Priority": "u=1",
    "Cache-Control": "no-cache",
}
browser_session = requests.Session()
browser_session.verify = False


def find_redirect_in_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    meta_refresh = soup.find("meta", attrs={"http-equiv": "refresh"})
    if meta_refresh:
        redirect_url = re.split(r";\s*url=", meta_refresh["content"])
        if len(redirect_url) > 1:
            return redirect_url[1].strip().strip("'\"")
    return None


def is_pdf(url):
    url_safe = url.split("?")[0].split("#")[0]
    if url_safe.endswith(".pdf"):
        return True
    response = browser_session.head(url, headers=browser_request_headers, allow_redirects=True)
    content_type = response.headers.get("content-type")
    if content_type and "pdf" in content_type.lower():
        return True
    return False


def is_xlsx(url):
    url_safe = url.split("?")[0].split("#")[0]
    if url_safe.endswith(".xlsx"):
        return True
    response = browser_session.head(url, headers=browser_request_headers, allow_redirects=True)
    content_type = response.headers.get("content-type")
    if content_type and "vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type.lower():
        return True
    return False


def is_docx(url):
    url_safe = url.split("?")[0].split("#")[0]
    if url_safe.endswith(".docx"):
        return True
    response = browser_session.head(url, headers=browser_request_headers, allow_redirects=True)
    content_type = response.headers.get("content-type")
    if content_type and "vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type.lower():
        return True
    return False


def extract_text_from_url(url, label="", redirect_list=[]):
    if label == "":
        label = url
    if os.path.isfile(url):
        ext = os.path.splitext(url)[1].lower()
        if ext == ".pdf":
            return extract_text_from_pdf(url, label=label)
        elif ext == ".xlsx":
            return extract_text_from_xlsx(url, label=label)
        elif ext == ".docx":
            return extract_text_from_docx(url, label=label)
        elif ext in (".html", ".htm"):
            return extract_text_from_html(url, label=label, redirect_list=redirect_list)
    if is_pdf(url):
        return extract_text_from_pdf(url, label=label)
    elif is_xlsx(url):
        return extract_text_from_xlsx(url, label=label)
    elif is_docx(url):
        return extract_text_from_docx(url, label=label)
    return extract_text_from_html(url, label=label, redirect_list=redirect_list)


def extract_text_from_html(url, label="", redirect_list=[]):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    html_file_path = dir_path + fs_safe_url(label) + ".html"
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    if os.path.exists(text_file_path):
        with open(text_file_path, "r") as file:
            text_content = file.read()
            if text_content:
                return text_content

    content = None
    if os.path.exists(html_file_path):
        with open(html_file_path, "rb") as file:
            content = file.read()
        # guard against poisoned caches (e.g. compressed bytes saved as HTML)
        if content and b"<" not in content[:4096]:
            print("Cached HTML is not decodable; refetching: " + url)
            content = None
    elif os.path.isfile(url):
        shutil.copyfile(url, html_file_path)
        with open(html_file_path, "rb") as file:
            content = file.read()
    else:
        try:
            response = browser_session.head(url, headers=browser_request_headers, allow_redirects=True)
            if response.status_code == 200:
                content_type = response.headers.get("content-type")
                if content_type and "html" in content_type.lower():
                    html_response = browser_session.get(url, headers=browser_request_headers, allow_redirects=True)
                    if html_response.status_code == 200:
                        content = html_response.content
                else:
                    print("Refusing to download unknown content type: " + content_type.lower())
                    return ""
            else:
                print("Failed to download:\n    " + str(response.status_code))
                return ""
        except Exception as e:
            print("Failed to download HTML")
            print(e)
            return ""

    if not content:
        print("Missing content: " + url)
        return ""

    redirect_url = find_redirect_in_html(content)
    if redirect_url:
        content = ""
        if redirect_url != url and redirect_url.startswith("http") and redirect_url not in redirect_list:
            redirect_list.append(redirect_url)
            return extract_text_from_url(redirect_url, label=label, redirect_list=redirect_list)
        else:
            print("Failed to retrieve redirect: " + url)
            return ""

    with open(html_file_path, "wb") as file:
        # raw extractions are build artifacts, not site pages: keep them out of the search index
        file.write(content.replace(b"<html", b"<html data-pagefind-ignore", 1))

    soup = BeautifulSoup(content, "html.parser")
    main_content = (
        soup.select_one("#main")
        or soup.find("main")
        or soup.find("article")
        or soup.select_one("#main-content")
        or soup.select_one("body > div.container")
        or soup.select_one("body")
    )
    if main_content:
        text_content = main_content.get_text(separator="\n", strip=True)
        with open(text_file_path, "w") as file:
            file.write(text_content)
        return text_content

    print("Main content not found: " + url)
    return ""


def extract_text_from_pdf(url, label=""):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    pdf_file_path = dir_path + fs_safe_url(label) + ".pdf"
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    if os.path.exists(text_file_path):
        with open(text_file_path, "r") as file:
            text_content = file.read()
            return text_content

    if not os.path.exists(pdf_file_path):
        if os.path.isfile(url):
            shutil.copyfile(url, pdf_file_path)
        elif url.startswith("http"):
            try:
                response = browser_session.get(url, headers=browser_request_headers, allow_redirects=True)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type")
                    if content_type and "pdf" in content_type.lower():
                        with open(pdf_file_path, "wb") as file:
                            file.write(response.content)
                    else:
                        soup = BeautifulSoup(response.content, "html.parser")
                        meta_refresh = soup.find("meta", attrs={"http-equiv": "refresh"})
                        if meta_refresh:
                            redirect_url = meta_refresh["content"].split(r";\s*url=")[1]
                            return extract_text_from_pdf(redirect_url, label)
                        else:
                            return "PDF file not found"
                else:
                    return "PDF file not downloaded"
            except Exception:
                print("PDF file not downloaded")
                return ""
        else:
            print("PDF file not found")
            return ""

    document = pymupdf.open(pdf_file_path)
    text_content = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text_content += page.get_text()

    if text_content:
        with open(text_file_path, "w") as file:
            file.write(text_content)
        return text_content
    else:
        print("PDF content not found")
        return ""


def extract_text_from_xlsx(url, label=""):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    xlsx_file_path = dir_path + fs_safe_url(label) + ".xlsx"
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    if os.path.exists(text_file_path):
        with open(text_file_path, "r") as file:
            text_content = file.read()
            return text_content

    if not os.path.exists(xlsx_file_path):
        if os.path.isfile(url):
            shutil.copyfile(url, xlsx_file_path)
        elif url.startswith("http"):
            try:
                response = browser_session.get(url, headers=browser_request_headers, allow_redirects=True)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type")
                    if content_type and "vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type.lower():
                        with open(xlsx_file_path, "wb") as file:
                            file.write(response.content)
                    else:
                        soup = BeautifulSoup(response.content, "html.parser")
                        meta_refresh = soup.find("meta", attrs={"http-equiv": "refresh"})
                        if meta_refresh:
                            redirect_url = meta_refresh["content"].split(r";\s*url=")[1]
                            return extract_text_from_xlsx(redirect_url, label)
                        else:
                            return "XLSX file not found"
                else:
                    return "XLSX file not downloaded"
            except Exception:
                print("XLSX file not downloaded")
                return ""
        else:
            print("XLSX file not found")
            return ""

    text_content = ""
    xls_data = pd.read_excel(xlsx_file_path, engine="openpyxl")
    xls_data.to_csv(text_file_path, index=False)

    with open(text_file_path, "r") as file:
        text_content = file.read()

    if text_content:
        return text_content
    else:
        print("XLSX content not found")
        return ""


def extract_text_from_docx(url, label=""):
    if label == "":
        label = url

    dir_path = "./sources/" + fs_safe_url(label) + "/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    docx_file_path = dir_path + fs_safe_url(label) + ".docx"
    text_file_path = dir_path + fs_safe_url(label) + ".txt"

    if os.path.exists(text_file_path):
        with open(text_file_path, "r") as file:
            text_content = file.read()
            return text_content

    if not os.path.exists(docx_file_path):
        if os.path.isfile(url):
            shutil.copyfile(url, docx_file_path)
        elif url.startswith("http"):
            try:
                response = browser_session.get(url, headers=browser_request_headers, allow_redirects=True)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type")
                    if content_type and (
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        in content_type.lower()
                    ):
                        with open(docx_file_path, "wb") as file:
                            file.write(response.content)
                    else:
                        soup = BeautifulSoup(response.content, "html.parser")
                        meta_refresh = soup.find("meta", attrs={"http-equiv": "refresh"})
                        if meta_refresh:
                            redirect_url = meta_refresh["content"].split(r";\s*url=")[1]
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

    doc = Document(docx_file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    text_content = "\n".join(text)

    if text_content:
        with open(text_file_path, "w") as file:
            file.write(text_content)
        return text_content
    else:
        print("DOCX content not found")
        return ""
