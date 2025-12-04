import sys
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich import box

encoder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
console = Console()


def sigmoid(x):
    """Converts a logit (e.g., 4.5) to a probability (0.0 - 1.0)"""
    return 1 / (1 + np.exp(-x))


def get_relevance_label(logit):
    """Maps a raw score to a human-readable label"""
    if logit >= 3:
        return "🟢"
    elif logit >= 0:
        return "🟡"
    return "🔴"


qdrant = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "3-2-1-newsletter"


@dataclass
class SemanticSearch:
    query: str


test_cases = [
    SemanticSearch(
        sys.argv[1] if len(sys.argv) > 1 else "The importance of priorities"
    ),
]

for case in test_cases:
    query = case.query

    # Calculate embeddings
    query_vector = encoder.encode(query, normalize_embeddings=True).tolist()

    # Initial retrieval
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=20
    ).points

    pairs = [[query, hit.payload.get("text", "")] for hit in hits]

    # Predict scores (This runs the heavy transformer math)
    cross_scores = reranker.predict(pairs)

    # Combine initial hits with new scores
    reranked_results = list(zip(hits, cross_scores))

    # Sort by the NEW score (Descending)
    reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)

    # Create Rich Table
    table = Table(title=f"Search Results for: '{query}'", box=box.ROUNDED)

    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Snippet", style="green")
    table.add_column("Init Score", justify="right")
    table.add_column("Rerank", justify="right")
    table.add_column("Prob", justify="right")
    table.add_column("Label", justify="center")

    for hit, new_score in reranked_results:
        date = hit.payload.get("date", "N/A")
        category = hit.payload.get("category", "N/A")
        text = hit.payload.get("text", "")
        prob = sigmoid(new_score) * 100
        label = get_relevance_label(new_score)

        # Truncate text for the table
        snippet = (text[:50] + "...") if len(text) > 50 else text

        table.add_row(
            str(date),
            str(category)[:15],
            snippet,
            f"{hit.score:.4f}",
            f"{new_score:+.4f}",
            f"{prob:3.1f}%",
            label,
        )

    console.print(table)
    console.print("\n")

    # Detailed view for top results (positive scores)
    for hit, new_score in reranked_results:
        if new_score >= 0:
            title = hit.payload.get("title", "No Title")
            date = hit.payload.get("date", "N/A")
            link = hit.payload.get("url", "#")
            text = hit.payload.get("text", "")

            console.print(f"[bold underline]{title}[/bold underline]")
            console.print(f"[dim]DATE: {date} | LINK: {link}[/dim]")
            console.print(f"{text}\n")
            console.print("---")