import os
import random
import re
import time
from datetime import datetime

import dagster as dg

from pipeline.utils import download_and_save, get_sitemap_urls

# Data directories
DATA_DIR = "data"
RAW_HTML_DIR = f"{DATA_DIR}/raw/html"

# Sitemap URL
SITEMAP_URL = "https://jamesclear.com/3-2-1-sitemap.xml"


@dg.asset(group_name="download_pipeline")
def sitemap_urls() -> list[str]:
    """Fetch sitemap URLs"""
    return get_sitemap_urls(SITEMAP_URL)


@dg.asset(group_name="download_pipeline")
def newsletter_issue_urls(
    sitemap_urls: list[str],
) -> dict[datetime, str]:
    """Parse sitemap URLs and extract newsletter issue dates"""
    issues: dict[datetime, str] = {}

    pattern = re.compile(
        r"https?://jamesclear\.com/3-2-1/(?P<month>[a-zA-Z]+)-(?P<day>\d{1,2})-(?P<year>\d{4})"
    )

    for url in sitemap_urls:
        match = pattern.search(url)

        if match:
            month_str = match.group("month")
            day_str = match.group("day")
            year_str = match.group("year")

            print(
                f"Raw Extraction: Month: {month_str}, Day: {day_str}, Year: {year_str}"
            )

            date_str = f"{month_str}-{day_str}-{year_str}"
            date_obj = datetime.strptime(date_str, "%B-%d-%Y")

            print(f"Date Object: {date_obj.date()}")

            issues[date_obj] = url

        else:
            print(f"Ignoring {url} - does not match pattern.")

    return issues


@dg.asset(group_name="download_pipeline")
def new_newsletter_urls(
    context: dg.AssetExecutionContext, newsletter_issue_urls: dict[datetime, str]
) -> list[str]:
    """Return all newsletter URLs sorted by date (filesystem is source of truth)"""
    # Sort URLs by date
    sorted_dict = dict(sorted(newsletter_issue_urls.items()))
    all_urls = list(sorted_dict.values())

    context.log.info(f"Found {len(all_urls)} newsletter URLs from sitemap")

    return all_urls


@dg.asset(group_name="download_pipeline")
def downloaded_html_files(
    context: dg.AssetExecutionContext, new_newsletter_urls: list[str]
) -> None:
    """Download newsletter HTML files to disk (skips existing files)"""
    os.makedirs(RAW_HTML_DIR, exist_ok=True)

    downloaded_count = 0
    skipped_count = 0

    for url in new_newsletter_urls:
        _, was_downloaded = download_and_save(url, RAW_HTML_DIR)

        if was_downloaded:
            downloaded_count += 1
            context.log.info(f"✓ Downloaded: {url}")

            # Only sleep after actual network requests
            sleep_time = random.uniform(1, 3)
            context.log.info(f"[SLEEP] Sleeping {sleep_time:.1f}s to be respectful...")
            time.sleep(sleep_time)
        else:
            skipped_count += 1
            context.log.info(f"⊘ Skipped (exists): {url}")

    context.log.info(f"Summary: {downloaded_count} downloaded, {skipped_count} skipped")

    return None
