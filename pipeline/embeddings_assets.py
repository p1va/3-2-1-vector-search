import os
from datetime import datetime
from pathlib import Path
from typing import Any

import dagster as dg
import pandas as pd

from pipeline.utils import parse_newsletter

# Data directories
DATA_DIR = "data"
RAW_MARKDOWN_DIR = f"{DATA_DIR}/raw/md"
PARQUET_DIR = f"{DATA_DIR}/parquet"


@dg.asset(group_name="embeddings_pipeline")
def text_chunks_for_embedding(
    context: dg.AssetExecutionContext, markdown_files: list[str]
) -> None:
    """Parse markdown files into text chunks for embedding"""
    os.makedirs(RAW_MARKDOWN_DIR, exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)

    all_issues = []

    for file in markdown_files:
        file_path = Path(file)

        context.log.info(f"Processing file: {file_path.name}")

        part = file_path.name.split(".md")[0].split("-")

        year = part[0]
        month = part[1]
        day = part[2]

        date_str = f"{year}-{month}-{day}"

        context.log.info(f"Date: {date_str}")

        date = datetime.strptime(date_str, "%Y-%m-%d")

        context.log.info(f"Preparing text to embedd for issue {date}")

        with open(
            file,
            "r",
            encoding="utf-8",
        ) as f:
            md_content = f.read()

        # Extract issue title from first line (should be # Title)
        title_content = "Unknown"
        lines = md_content.split("\n", 1)
        if lines and lines[0].startswith("# "):
            title_content = lines[0].replace("# ", "").strip()

        # Construct newsletter URL from date
        month_name = date.strftime("%B").lower()  # e.g., "april"
        day_num = date.strftime("%-d" if os.name != "nt" else "%#d")  # e.g., "10"
        newsletter_url = f"https://jamesclear.com/3-2-1/{month_name}-{day_num}-{year}"

        context.log.info(f"Title: {title_content}")
        context.log.info(f"URL: {newsletter_url}")

        # Extract content
        sections: list[dict[str, Any]] = parse_newsletter(
            md_content, issue_date=date_str
        )

        # Add title and URL to each chunk's metadata
        for section in sections:
            section["metadata"]["title"] = title_content
            section["metadata"]["url"] = newsletter_url

        all_issues.extend(sections)

    df = pd.json_normalize(all_issues)
    df.columns = [c.replace("metadata.", "") for c in df.columns]
    df.to_parquet(
        os.path.join(
            PARQUET_DIR,
            "newsletter_embeddings.parquet",
        ),
        engine="pyarrow",
        index=False,
    )

    return None
