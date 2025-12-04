import hashlib
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_sitemap_urls(sitemap_index_url: str) -> list[str]:
    """
    Parses the Sitemap Index to find the post-sitemap,
    then parses that to find 3-2-1 URLs.
    """
    print(f"Fetching sitemap index: {sitemap_index_url}")

    try:
        response = requests.get(sitemap_index_url, headers=HEADERS)

        if response.status_code != 200:
            print(f"Failed to get sitemap. Status: {response.status_code}")
            return []

        # Parse the XML
        root = ET.fromstring(response.content)

        ns = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        return list(
            loc.text
            for loc in root.findall(".//sitemap:loc", ns)
            if loc.text is not None
        )

    except Exception as e:
        print(f"Error parsing sitemap: {e}")
        return []


def get_safe_filename(url):
    slug = url.strip("/").split("/")[-1]
    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{slug}_{url_hash}.html"


def download_and_save(url, output_dir):
    """Download URL to disk. Returns (file_path, was_downloaded)"""
    filename = get_safe_filename(url)
    final_path = os.path.join(output_dir, filename)

    if os.path.exists(final_path):
        print(f"Skipping {url} (File exists)")
        return final_path, False

    print(f"Downloading {url}...")

    try:
        response = requests.get(url, timeout=10, headers=HEADERS)
        response.raise_for_status()

        # Atomic write pattern: first on temp then rename
        temp_path = final_path + ".tmp"

        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        os.rename(temp_path, final_path)

        return final_path, True

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        temp_path = final_path + ".tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def clean_links(text):
    """Regex to find [text](url) and replaces it with just 'text'"""
    return re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)


def trim_empty_lines(text: str):
    text = "\n".join([s for s in text.strip().splitlines() if s.strip()])
    return text.strip()


def parse_newsletter(md_text, issue_date="Unknown") -> list[dict[str, Any]]:
    """Parse newsletter markdown into structured chunks"""
    chunks: list[dict[str, Any]] = []

    clean_text = re.sub(r"^\[Share this on.*\n?", "", md_text, flags=re.MULTILINE)

    # Header pattern
    pattern = r"^##\s+"

    sections = re.split(pattern, clean_text, flags=re.MULTILINE)

    print(f"Total sections found: {len(sections)}")

    for section in sections:
        section = section.strip()
        section = section.replace("---", "")

        if "3 IDEAS FROM ME" in section:
            # Split by Roman Numerals (I., II., III.)
            ideas = re.split(r"[IVX]+\.", section, flags=re.MULTILINE)

            for i, idea in enumerate(ideas[1:], 1):  # Skip header
                chunks.append(
                    {
                        "text": "Idea from James Clear: " + trim_empty_lines(idea),
                        "metadata": {
                            "category": "idea",
                            "index": i,
                            "date": issue_date,
                        },
                    }
                )

        elif "2 QUOTES FROM OTHERS" in section:
            quotes = re.split(r"[IVX]+\.", section, flags=re.MULTILINE)
            for i, quote in enumerate(quotes[1:], 1):
                source_match = re.search(
                    r"\*Source:\*\s*\[([^\]]+)\]\(([^\)]+)\)", quote
                )

                source_title = None
                source_url = None

                if source_match:
                    # We found a link! Capture title and URL
                    source_title = source_match.group(1).replace("*", "")
                    source_url = source_match.group(2)
                else:
                    # Fallback: Maybe there was no link, just text
                    text_match = re.search(r"\*Source:\*\s*(.+)$", quote, re.MULTILINE)

                    if text_match:
                        source_title = text_match.group(1).replace("*", "")

                clean_quote = re.sub(r"\n\*Source:\*.*", "", quote, flags=re.DOTALL)
                clean_quote = clean_links(clean_quote)
                clean_quote = clean_quote.replace("**", "").replace("  ", " ").strip()

                # Only add "Quote from {source}" prefix if source is not None
                if source_title:
                    final_quote = f"Quote from {source_title}: {clean_quote}"
                else:
                    final_quote = clean_quote

                chunks.append(
                    {
                        "text": trim_empty_lines(final_quote),
                        "metadata": {
                            "category": "quote",
                            "source": source_url,
                            "source_name": source_title,
                            "index": i,
                            "date": issue_date,
                        },
                    }
                )

        elif "1 QUESTION FOR YOU" in section:
            # Usually just one block of text after the header
            question_text = section.replace("1 QUESTION FOR YOU", "").strip()

            question_text = question_text.split("Until next week")[0].strip()

            chunks.append(
                {
                    "text": trim_empty_lines(question_text),
                    "metadata": {
                        "category": "question",
                        "index": 1,
                        "date": issue_date,
                    },
                }
            )

    return chunks
