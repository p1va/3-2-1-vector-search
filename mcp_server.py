"""
MCP Server for 3-2-1 Newsletter Semantic Search

This server provides a tool to search through James Clear's newsletter archive
using semantic search with reranking.
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer

# Constants
COLLECTION_NAME = "3-2-1-newsletter"
PARQUET_PATH = "data/parquet/newsletter_embeddings.parquet"


# Global variables for models and database
encoder: SentenceTransformer
reranker: CrossEncoder
qdrant: QdrantClient


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Initialize models and database on server startup"""
    global encoder, reranker, qdrant

    # Load models
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Initialize In-Memory Qdrant
    qdrant = QdrantClient(":memory:")

    # Load Data
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Parquet file not found at {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH)

    # Create Collection
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.DOT),
    )

    # Upload Points
    points = []
    for _, row in df.iterrows():
        vector = row["vector"].tolist()
        payload = row.to_dict()
        payload.pop("vector", None)

        points.append(
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload=payload,
            )
        )

    qdrant.upload_points(collection_name=COLLECTION_NAME, points=points, batch_size=64)

    yield  # Server runs here


# Initialize FastMCP server
mcp = FastMCP("3-2-1 Newsletter Search", lifespan=server_lifespan)


@mcp.tool()
def search_newsletter(
    query: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    min_score: float = 0.0,
    limit: int = 10,
) -> dict:
    """
    Search through the 3-2-1 newsletter archive using semantic search.

    Args:
        query: The search query text (e.g., "the importance of picking the right priorities")
        from_date: Optional start date filter in YYYY-MM-DD format (e.g., "2019-10-10")
        to_date: Optional end date filter in YYYY-MM-DD format (e.g., "2023-12-31")
        min_score: Minimum relevance score threshold (default: 0.0). Higher means more relevant.
        limit: Maximum number of results to return (default: 10)

    Returns:
        A dictionary with query info and results list containing title, date, category, URL, text, and relevance score
    """
    # Parse dates if provided
    from_datetime = None
    to_datetime = None
    if from_date:
        try:
            from_datetime = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            return {
                "error": "Invalid from_date format. Use YYYY-MM-DD (e.g., '2019-10-10')"
            }

    if to_date:
        try:
            to_datetime = datetime.strptime(to_date, "%Y-%m-%d")
        except ValueError:
            return {
                "error": "Invalid to_date format. Use YYYY-MM-DD (e.g., '2023-12-31')"
            }

    # Encode query
    query_vector = encoder.encode(query, normalize_embeddings=True).tolist()

    # Initial retrieval (get more to account for filtering)
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=50
    ).points

    if not hits:
        return {
            "query": query,
            "filters": {
                "from_date": from_date,
                "to_date": to_date,
                "min_score": min_score,
            },
            "total_results": 0,
            "results": [],
        }

    # Prepare pairs for reranking
    pairs = [[query, hit.payload.get("text", "")] for hit in hits]

    # Rerank
    cross_scores = reranker.predict(pairs)

    # Combine hits with scores
    reranked_results = list(zip(hits, cross_scores))

    # Sort by new score (descending)
    reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)

    # Filter by date and score
    filtered_results = []
    for hit, score in reranked_results:
        # Filter by minimum score
        if score < min_score:
            continue

        # Filter by date
        date_str = hit.payload.get("date")
        if date_str:
            try:
                # Parse date (handle different formats if needed)
                result_date = datetime.strptime(date_str, "%Y-%m-%d")

                if from_datetime and result_date < from_datetime:
                    continue
                if to_datetime and result_date > to_datetime:
                    continue
            except ValueError:
                # If date parsing fails, skip date filtering for this item
                pass

        filtered_results.append((hit, score))

        # Stop if we've collected enough results
        if len(filtered_results) >= limit:
            break

    if not filtered_results:
        return {
            "query": query,
            "filters": {
                "from_date": from_date,
                "to_date": to_date,
                "min_score": min_score,
            },
            "total_results": 0,
            "results": [],
        }

    # Build structured results
    results = []
    for hit, score in filtered_results:
        payload = hit.payload
        results.append(
            {
                "title": str(payload.get("title", "")),
                "date": str(payload.get("date", "")),
                "category": str(payload.get("category", "")),
                "url": str(payload.get("url", "")),
                "text": str(payload.get("text", "")),
                "score": round(float(score), 4),
            }
        )

    return {
        "query": query,
        "filters": {
            "from_date": from_date,
            "to_date": to_date,
            "min_score": min_score,
        },
        "total_results": len(results),
        "results": results,
    }


if __name__ == "__main__":
    mcp.run(show_banner=False)
