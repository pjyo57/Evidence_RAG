from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Embedding parameters
# -----------------------------

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(chunks_file):
    """Load chunks from the processed JSON file."""

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


def create_embeddings(chunks, model):
    """Create one embedding vector for each chunk."""

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":

    # Find project root automatically
    project_root = Path(__file__).resolve().parent.parent

    processed_dir = project_root / "data" / "processed"

    chunks_file = processed_dir / "chunks.json"
    embeddings_file = processed_dir / "embeddings.npy"

    # -----------------------------
    # Load chunks
    # -----------------------------

    chunks = load_chunks(chunks_file)

    print("--------------------------------")
    print("LOADED CHUNKS")
    print("--------------------------------")

    print(f"Number of chunks: {len(chunks)}")

    # -----------------------------
    # Load embedding model
    # -----------------------------

    print("\n--------------------------------")
    print("LOADING EMBEDDING MODEL")
    print("--------------------------------")

    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # -----------------------------
    # Create embeddings
    # -----------------------------

    print("\n--------------------------------")
    print("CREATING EMBEDDINGS")
    print("--------------------------------")

    embeddings = create_embeddings(chunks, model)

    print("\nEmbedding creation complete.")

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding data type: {embeddings.dtype}")

    # -----------------------------
    # Save embeddings
    # -----------------------------

    np.save(embeddings_file, embeddings)

    print("\n--------------------------------")
    print("EMBEDDINGS SAVED")
    print("--------------------------------")

    print(f"File: {embeddings_file}")
    print(f"Shape: {embeddings.shape}")

    # -----------------------------
    # Inspect first embedding
    # -----------------------------

    print("\n--------------------------------")
    print("FIRST EMBEDDING")
    print("--------------------------------")

    print(embeddings[0][:10])