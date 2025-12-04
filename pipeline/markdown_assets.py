import os
from datetime import datetime
from pathlib import Path

import dagster as dg
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Data directories
DATA_DIR = "data"
RAW_HTML_DIR = f"{DATA_DIR}/raw/html"
RAW_MARKDOWN_DIR = f"{DATA_DIR}/raw/md"


@dg.asset(group_name="markdown_pipeline")
def html_files() -> list[str]:
    """Scan and return all HTML files from disk"""
    os.makedirs(RAW_HTML_DIR, exist_ok=True)

    files: list[str] = []

    for subdir, dirs, filenames in os.walk(RAW_HTML_DIR):
        for filename in filenames:
            if filename.endswith((".html")):
                files.append(os.path.join(subdir, filename))

    return files


@dg.asset(group_name="markdown_pipeline")
def markdown_files(
    context: dg.AssetExecutionContext, html_files: list[str]
) -> list[str]:
    """Convert HTML files to Markdown and return MD file paths"""
    os.makedirs(RAW_MARKDOWN_DIR, exist_ok=True)

    md_files: list[str] = []

    for file in html_files:
        file_path = Path(file)

        context.log.info(f"Processing file: {file_path.name}")

        part = file_path.name.split("_")[0].split("-")

        month_str = part[0]
        day_str = part[1]
        year_str = part[2]

        date_str = f"{month_str}-{day_str}-{year_str}"

        context.log.info(f"Date: {date_str}")

        date_obj = datetime.strptime(date_str, "%B-%d-%Y")

        context.log.info(f"Parsing HTML to MD for issue {date_obj}")

        with open(
            file,
            "r",
            encoding="utf-8",
        ) as f:
            html_content = f.read()

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract issue title
        title_content = soup.select_one(".page__header h1")

        if title_content:
            title_content = title_content.get_text(strip=True)

        # Extract content
        main_content = soup.find("div", {"class": "page__content"})

        if main_content:
            # Convert HTML to markdown
            markdown_text = md(str(main_content), heading_style="ATX")

            # Atomic write pattern: first on temp then rename
            final_path = os.path.join(
                RAW_MARKDOWN_DIR, f"{date_obj.strftime('%Y-%m-%d')}.md"
            )

            temp_path = final_path + ".tmp"

            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(f"# {title_content}\n\n" + markdown_text)

            os.rename(temp_path, final_path)
            md_files.append(final_path)

    return md_files
