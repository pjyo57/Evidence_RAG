from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------
# PARAMETERS
# --------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

# Penalty applied to reference-like chunks.
# This is our experimental parameter.
REFERENCE_PENALTY = 0.10


# --------------------------------
# LOAD DATA
# --------------------------------

def load_chunks(chunks_file):
    with open(
        chunks_file,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    return chunks


def load_embeddings(embeddings_file):

    embeddings = np.load(
        embeddings_file
    )

    return embeddings


# --------------------------------
# COSINE SIMILARITY
# --------------------------------

def cosine_similarity(
    query_embedding,
    document_embeddings
):
    """
    Calculate cosine similarity between
    the query and every document chunk.
    """

    query_norm = np.linalg.norm(
        query_embedding
    )

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1
    )

    similarities = (
        document_embeddings @ query_embedding
        / (document_norms * query_norm)
    )

    return similarities


# --------------------------------
# REFERENCE DETECTION
# --------------------------------

def looks_like_reference_chunk(text):
    """
    Conservative heuristic for identifying
    bibliography-like chunks.

    We require multiple citation-style
    patterns before classifying a chunk
    as reference-like.
    """

    import re

    citation_patterns = [
        r"\b\d{4}\.",
        r"\bet al\.",
        r"\bProceedings of\b",
        r"\barXiv preprint\b",
        r"\bdoi\.org\b",
    ]

    matches = 0

    for pattern in citation_patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            matches += 1

    return matches >= 3


# --------------------------------
# REFERENCE-AWARE RETRIEVAL
# --------------------------------

def retrieve_reference_aware(
    question,
    chunks,
    embeddings,
    model,
    top_k=TOP_K,
    reference_penalty=REFERENCE_PENALTY
):
    """
    Retrieve chunks using cosine similarity,
    while penalizing chunks that appear to be
    bibliography/reference material.

    Original similarity is preserved for analysis.
    """

    query_embedding = model.encode(
        question,
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )

    adjusted_scores = similarities.copy()

    reference_flags = []

    for i, chunk in enumerate(chunks):

        is_reference = (
            looks_like_reference_chunk(
                chunk["text"]
            )
        )

        reference_flags.append(
            is_reference
        )

        if is_reference:

            adjusted_scores[i] -= (
                reference_penalty
            )

    ranked_indices = np.argsort(
        adjusted_scores
    )[::-1]

    top_indices = ranked_indices[:top_k]

    retrieved_chunks = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        chunk = chunks[index].copy()

        chunk["rank"] = rank

        chunk["similarity"] = float(
            similarities[index]
        )

        chunk["adjusted_similarity"] = float(
            adjusted_scores[index]
        )

        chunk["reference_like"] = (
            reference_flags[index]
        )

        retrieved_chunks.append(
            chunk
        )

    return retrieved_chunks


# --------------------------------
# MAIN EXPERIMENT
# --------------------------------

if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    processed_dir = (
        project_root
        / "data"
        / "processed"
    )

    experiments_dir = (
        project_root
        / "experiments"
    )

    chunks_file = (
        processed_dir
        / "chunks.json"
    )

    embeddings_file = (
        processed_dir
        / "embeddings.npy"
    )

    questions_file = (
        experiments_dir
        / "questions.json"
    )


    # --------------------------------
    # LOAD DATA
    # --------------------------------

    print("--------------------------------")
    print("LOADING DATA")
    print("--------------------------------")

    chunks = load_chunks(
        chunks_file
    )

    embeddings = load_embeddings(
        embeddings_file
    )

    with open(
        questions_file,
        "r",
        encoding="utf-8"
    ) as f:

        questions = json.load(f)


    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Embeddings: {embeddings.shape}"
    )


    # --------------------------------
    # LOAD MODEL
    # --------------------------------

    print("\n--------------------------------")
    print("LOADING EMBEDDING MODEL")
    print("--------------------------------")

    print(
        f"Model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


    # --------------------------------
    # RUN EXPERIMENT
    # --------------------------------

    print("\n================================")
    print("REFERENCE-AWARE RETRIEVAL")
    print("================================")

    print(
        f"Top-k: {TOP_K}"
    )

    print(
        f"Reference penalty: "
        f"{REFERENCE_PENALTY}"
    )


    all_results = []

    for question_data in questions:

        question_id = question_data["id"]

        question = question_data[
            "question"
        ]

        print("\n--------------------------------")
        print(question_id)
        print("--------------------------------")

        print(
            f"Question: {question}"
        )

        retrieved_chunks = (
            retrieve_reference_aware(
                question,
                chunks,
                embeddings,
                model
            )
        )


        reference_count = sum(
            chunk["reference_like"]
            for chunk in retrieved_chunks
        )


        print(
            f"Reference-like chunks in "
            f"top-{TOP_K}: "
            f"{reference_count}"
        )


        for chunk in retrieved_chunks:

            print(
                f"\nRank {chunk['rank']}"
            )

            print(
                f"File: "
                f"{chunk['filename']}"
            )

            print(
                f"Page: "
                f"{chunk['page_number']}"
            )

            print(
                f"Original similarity: "
                f"{chunk['similarity']:.4f}"
            )

            print(
                f"Adjusted similarity: "
                f"{chunk['adjusted_similarity']:.4f}"
            )

            print(
                f"Reference-like: "
                f"{chunk['reference_like']}"
            )

            print(
                f"Text: "
                f"{chunk['text'][:300]}"
            )


        all_results.append({
            "id": question_id,
            "question": question,
            "type": question_data["type"],
            "answerable": question_data["answerable"],
            "expected_sources": question_data[
                "expected_sources"
            ],
            "retrieved_chunks": retrieved_chunks,
            "reference_like_count": reference_count
        })


    # --------------------------------
    # SAVE RESULTS
    # --------------------------------

    output_file = (
        experiments_dir
        / "reference_aware_retrieval_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------
    # SUMMARY
    # --------------------------------

    total_retrieved = (
        len(all_results) * TOP_K
    )

    total_reference_like = sum(
        result["reference_like_count"]
        for result in all_results
    )

    percentage = (
        total_reference_like
        / total_retrieved
        * 100
    )


    print("\n================================")
    print("REFERENCE-AWARE RETRIEVAL COMPLETE")
    print("================================")

    print(
        f"Total retrieved chunks: "
        f"{total_retrieved}"
    )

    print(
        f"Reference-like retrieved chunks: "
        f"{total_reference_like}"
    )

    print(
        f"Percentage reference-like: "
        f"{percentage:.2f}%"
    )

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )