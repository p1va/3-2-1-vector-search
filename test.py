import re
from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify as md


def clean_links(text):
    # Regex to find [text](url) and replaces it with just "text"
    return re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)


def trim_empty_lines(text: str):
    text = "\n".join([s for s in text.strip().splitlines() if s.strip()])
    return text.strip()


def parse_newsletter(md_text, issue_date="Unknown"):
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
                            "category": "Idea",
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

                final_quote = f"Quote from {source_title} {clean_quote}"

                # Pull source out

                chunks.append(
                    {
                        "text": trim_empty_lines(final_quote),
                        "metadata": {
                            "category": "Quote",
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
                        "category": "Question",
                        "index": 1,
                        "date": issue_date,
                    },
                }
            )

    return chunks


# 1. Load your HTML file
with open(
    # "src/321_vector_store/defs/data/raw/html/april-7-2022_5bba52.html",
    "src/321_vector_store/defs/data/raw/html/april-3-2025_bfbf93.html",
    "r",
    encoding="utf-8",
) as f:
    html_content = f.read()

# Parse HTML to find the specific content area
soup = BeautifulSoup(html_content, "html.parser")

title_content = soup.select_one(".page__header h1")

if title_content:
    print(f"{title_content.get_text(strip=True)}")

# Change 'main' or 'id="content"' to match your specific HTML structure
main_content = soup.find("div", {"class": "page__content"})

# Convert only that section
if main_content:
    markdown_text = md(str(main_content), heading_style="ATX")

    print(markdown_text)

    sections = parse_newsletter(markdown_text, issue_date="Oct 12, 2023")

    print("$$$")
    print("$$$ Sections")
    print("$$$")

    for section in sections:
        print(
            f"--- {section['metadata']['category']} #{section['metadata']['index']} ---"
        )
        print(section["text"])
        print("---")
