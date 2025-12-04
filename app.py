import os
import uuid

import numpy as np
import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer

# Page Config
st.set_page_config(page_title="3-2-1 Newsletter Search", page_icon="📚", layout="wide")

# Constants
COLLECTION_NAME = "3-2-1-newsletter"
PARQUET_PATH = "data/parquet/newsletter_embeddings.parquet"


@st.cache_resource
def load_models():
    """Load models once and cache them to avoid reloading on every interaction."""
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return encoder, reranker


@st.cache_resource
def init_vector_db():
    """
    Initialize an in-memory Qdrant instance and load data from Parquet.
    This runs only once when the app starts.
    """
    # 1. Initialize In-Memory Qdrant
    qdrant = QdrantClient(":memory:")

    # 2. Load Data
    if not os.path.exists(PARQUET_PATH):
        return None

    df = pd.read_parquet(PARQUET_PATH)

    # 3. Create Collection
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.DOT),
    )

    # 4. Upload Points
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

    return qdrant


@st.cache_data
def perform_search(query_text):
    """
    Executes the search and reranking.
    Cached so repeating the same query is instant.
    """
    # Retrieve cached resources
    encoder, reranker = load_models()
    qdrant = init_vector_db()

    if not qdrant:
        return None

    # 1. Vector Search (Retrieval)
    query_vector = encoder.encode(query_text, normalize_embeddings=True).tolist()

    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=20
    ).points

    if not hits:
        return []

    # 2. Reranking
    pairs = [[query_text, hit.payload.get("text", "")] for hit in hits]
    cross_scores = reranker.predict(pairs)

    reranked_results = list(zip(hits, cross_scores))
    # Sort by new score
    reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)

    return reranked_results


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_relevance_label(logit):
    if logit >= 3:
        return "🟢 High"
    elif logit >= 0:
        return "🟡 Medium"
    return "🔴 Low"


# --- UI Structure ---

st.title("📚 3-2-1 Newsletter Semantic Search")
st.markdown("""
Search through James Clear's **3-2-1 Newsletter** archive.  
This app uses Vector search + Reranking to find relevant issues.
""")

# Ensure data exists
if not os.path.exists(PARQUET_PATH):
    st.error(
        f"⚠️ Data file not found at `{PARQUET_PATH}`. Please run the ingestion pipeline first."
    )
    st.stop()

# Load Resources (Trigger cache)
with st.spinner("Booting up knowledge base..."):
    load_models()
    init_vector_db()

# Search UI
with st.form("search_form"):
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g., The importance of picking the right priorities",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Search", use_container_width=True)

if submitted and query:
    with st.spinner(f"Searching for '{query}'..."):
        results = perform_search(query)

    if results is None:
        st.error("Database connection failed.")
    elif results:
        st.write(f"Found **{len(results)}** results.")
        st.divider()

        for hit, new_score in results:
            # Threshold for display
            if new_score > -2:
                payload = hit.payload
                date = payload.get("date", "Unknown Date")
                category = payload.get("category", "General")
                text = payload.get("text", "")
                url = payload.get("url", "#")
                title = payload.get("title", "Newsletter Issue")

                score_label = get_relevance_label(new_score)
                prob = sigmoid(new_score) * 100

                with st.container():
                    c1, c2 = st.columns([0.85, 0.15])
                    with c1:
                        st.markdown(f"### [{title}]({url})")
                        st.caption(f"📅 {date} | 🏷️ {category}")
                        st.markdown(f"> {text}")
                    with c2:
                        st.metric("Relevance", f"{prob:.0f}%", delta=score_label)

                    st.divider()
    else:
        st.info("No relevant results found.")

elif not query and submitted:
    st.warning("Please enter a search term.")
