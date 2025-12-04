import os

import dagster as dg
import pandas as pd
from sentence_transformers import SentenceTransformer

# Data directories
DATA_DIR = "data"
RAW_MARKDOWN_DIR = f"{DATA_DIR}/raw/md"
PARQUET_DIR = f"{DATA_DIR}/parquet"


@dg.asset(group_name="encode_pipeline", deps=["text_chunks_for_embedding"])
def encoded_vectors(context: dg.AssetExecutionContext) -> None:
    """Encode text chunks into vectors"""
    os.makedirs(PARQUET_DIR, exist_ok=True)

    parquet_file = os.path.join(PARQUET_DIR, "newsletter_embeddings.parquet")

    df = pd.read_parquet(parquet_file)

    context.log.info("Loading embedding model...")

    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

    context.log.info("Encoding vectors...")

    vectors = embeddings_model.encode(
        df["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    df["vector"] = vectors.tolist()

    # Save to Parquet
    context.log.info("Updating parquet file with vectors..")

    df.to_parquet(parquet_file)

    context.log.info("Done")
