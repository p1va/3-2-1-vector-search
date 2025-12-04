import os
import uuid

import dagster as dg
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Data directories
DATA_DIR = "data"

PARQUET_DIR = f"{DATA_DIR}/parquet"
COLLECTION_NAME = "3-2-1-newsletter"


@dg.asset(group_name="store_pipeline", deps=["encoded_vectors"])
def stored_vectors(context: dg.AssetExecutionContext) -> None:
    """Store vectors in Qdrant"""
    os.makedirs(PARQUET_DIR, exist_ok=True)

    parquet_file = os.path.join(PARQUET_DIR, "newsletter_embeddings.parquet")

    df = pd.read_parquet(parquet_file)

    context.log.info(
        f"Loaded {len(df)} records. Vector shape: {len(df.iloc[0]['vector'])}"
    )

    qdrant = QdrantClient(url="http://localhost:6333")

    try:
        qdrant.get_collections()
        context.log.info("✅ Qdrant is reachable!")
    except Exception as e:
        context.log.info(f"❌ Could not connect to Qdrant: {e}")
        raise e

    if not qdrant.collection_exists(COLLECTION_NAME):
        context.log.info(f"Creating collection '{COLLECTION_NAME}'...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.DOT),
        )
    else:
        context.log.info(
            f"Collection '{COLLECTION_NAME}' already exists. Recreating..."
        )
        qdrant.delete_collection(COLLECTION_NAME)
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.DOT),
        )

    points: list[PointStruct] = []

    for i, (idx, row) in enumerate(df.iterrows()):
        # Pull the vector out into a list
        vector: list[float] = row["vector"].tolist()

        # Convert the row to a dictionary and remove vector
        payload = row.to_dict()
        payload.pop("vector", None)

        points.append(
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload=payload,
            )
        )

    qdrant.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=64,
        parallel=1,
    )

    context.log.info(f"Success! {len(points)} records indexed.")
