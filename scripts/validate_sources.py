"""Validate all URLs in config/sources.csv — checks for HTTP 200 and flags archived URLs."""

import csv
import sys
from collections import Counter
from urllib.parse import urlparse

import requests

ARCHIVE_DOMAINS = {"obamawhitehouse.archives.gov", "georgewbush-whitehouse.archives.gov"}


def read_sources(path):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)
        for row in reader:
            cleaned = [field.strip().strip('"') for field in row]
            if any(cleaned):
                yield cleaned


def check_url(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=15, headers={"User-Agent": "govdoc-explainer/2.0"})
        return response.status_code, response.url
    except requests.RequestException as e:
        return None, str(e)


def main():
    path = "config/sources.csv"
    ok = 0
    broken = 0
    archived = 0
    status_codes = Counter()

    rows = list(read_sources(path))

    for row in rows:
        if len(row) < 3:
            continue
        category, standard, url = row[0], row[1], row[2]
        if not url:
            continue

        domain = urlparse(url).netloc
        is_archive = domain in ARCHIVE_DOMAINS
        status, detail = check_url(url)

        if status == 200:
            ok += 1
            if is_archive:
                archived += 1
                print(f"  [ARCHIVE] {standard[:80]}")
        else:
            broken += 1
            print(f"  [BROKEN {status}] {standard[:80]}")
            print(f"    {url}")
            if detail and detail != url:
                print(f"    -> {detail[:100]}")

        if status:
            status_codes[status] += 1

    print(f"\n{'='*60}")
    print(f"Total: {len(rows)} sources")
    print(f"  OK:      {ok}")
    print(f"  Broken:  {broken}")
    print(f"  Archive: {archived}")
    print(f"  Status codes: {dict(status_codes)}")

    return 1 if broken > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
