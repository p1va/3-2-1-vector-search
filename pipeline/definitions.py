from dagster import (
    AssetSelection,
    Definitions,
    define_asset_job,
    load_assets_from_modules,
)

from pipeline import embeddings_assets, encode_assets, store_assets

from . import download_assets, markdown_assets

# Load all assets
all_assets = load_assets_from_modules(
    [download_assets, markdown_assets, embeddings_assets, encode_assets, store_assets]
)

# Define jobs for each pipeline
download_job = define_asset_job(
    name="download_pipeline",
    selection=AssetSelection.groups("download_pipeline"),
    description="Download newsletter issues from sitemap to HTML files",
)

markdown_job = define_asset_job(
    name="markdown_pipeline",
    selection=AssetSelection.groups("markdown_pipeline"),
    description="Process HTML files to Markdown and chunks",
)

embeddings_job = define_asset_job(
    name="embeddings_pipeline",
    selection=AssetSelection.groups("embeddings_pipeline"),
    description="Embeddings from Markdown",
)

encode_job = define_asset_job(
    name="encode_pipeline",
    selection=AssetSelection.groups("encode_pipeline"),
    description="Encoding text into vectors",
)

store_job = define_asset_job(
    name="store_pipeline",
    selection=AssetSelection.groups("store_pipeline"),
    description="Store vectors into Qdrant",
)

# Combined job: runs markdown → embeddings → encode → store
full_processing_job = define_asset_job(
    name="full_processing_pipeline",
    selection=AssetSelection.groups(
        "markdown_pipeline", "embeddings_pipeline", "encode_pipeline", "store_pipeline"
    ),
    description="Full pipeline: HTML → Markdown → Text chunks → Vectors → Qdrant",
)

defs = Definitions(
    assets=all_assets,
    jobs=[
        download_job,
        markdown_job,
        embeddings_job,
        encode_job,
        store_job,
        full_processing_job,
    ],
)
