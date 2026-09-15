from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Retrieval parameters
# -----------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def load_chunks(chunks_file):
    """Load chunk metadata and text."""

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


def load_embeddings(embeddings_file):
    """Load the stored chunk embeddings."""

    return np.load(embeddings_file)


def load_questions(questions_file):
    """Load evaluation questions."""

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    return questions


def cosine_similarity(query_embedding, document_embeddings):
    """
    Calculate cosine similarity between one query vector
    and all document/chunk vectors.
    """

    query_norm = np.linalg.norm(query_embedding)

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1
    )

    similarities = (
        document_embeddings @ query_embedding
        / (document_norms * query_norm)
    )

    return similarities


def retrieve(query, model, chunks, embeddings, top_k=TOP_K):
    """
    Retrieve the top-k most similar chunks for a query.
    """

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:

        result = {
            "chunk_id": chunks[index]["chunk_id"],
            "filename": chunks[index]["filename"],
            "page_number": chunks[index]["page_number"],
            "chunk_number": chunks[index]["chunk_number"],
            "similarity": float(similarities[index]),
            "text": chunks[index]["text"]
        }

        results.append(result)

    return results


def evaluate_retrieval(questions, model, chunks, embeddings, top_k=TOP_K):
    """
    Evaluate whether an expected source document appears
    in the top-k retrieved chunks.
    """

    results = []

    for question_data in questions:

        question = question_data["question"]
        expected_sources = question_data["expected_sources"]

        retrieved = retrieve(
            query=question,
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=top_k
        )

        retrieved_sources = [
            result["filename"]
            for result in retrieved
        ]

        # For answerable questions, check whether
        # at least one expected source was retrieved.
        if question_data["answerable"]:

            retrieval_hit = any(
                source in retrieved_sources
                for source in expected_sources
            )

        else:
            # Unanswerable questions do not have an expected
            # source. We record them separately.
            retrieval_hit = None

        result = {
            "id": question_data["id"],
            "question": question,
            "type": question_data["type"],
            "answerable": question_data["answerable"],
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "retrieval_hit": retrieval_hit,
            "top_results": retrieved
        }

        results.append(result)

    return results


if __name__ == "__main__":

    # Find project root automatically
    project_root = Path(__file__).resolve().parent.parent

    processed_dir = project_root / "data" / "processed"
    experiments_dir = project_root / "experiments"

    chunks_file = processed_dir / "chunks.json"
    embeddings_file = processed_dir / "embeddings.npy"
    questions_file = experiments_dir / "questions.json"

    results_file = experiments_dir / "retrieval_results.json"

    # -----------------------------
    # Load data
    # -----------------------------

    print("--------------------------------")
    print("LOADING RETRIEVAL DATA")
    print("--------------------------------")

    chunks = load_chunks(chunks_file)
    embeddings = load_embeddings(embeddings_file)
    questions = load_questions(questions_file)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Number of questions: {len(questions)}")

    # Make sure chunks and embeddings correspond
    assert len(chunks) == len(embeddings), (
        "Number of chunks and embeddings do not match."
    )

    # -----------------------------
    # Load embedding model
    # -----------------------------

    print("\n--------------------------------")
    print("LOADING EMBEDDING MODEL")
    print("--------------------------------")

    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # -----------------------------
    # Evaluate retrieval
    # -----------------------------

    print("\n--------------------------------")
    print("RUNNING RETRIEVAL EVALUATION")
    print("--------------------------------")

    results = evaluate_retrieval(
        questions=questions,
        model=model,
        chunks=chunks,
        embeddings=embeddings,
        top_k=TOP_K
    )

    # -----------------------------
    # Calculate Retrieval@K
    # -----------------------------

    answerable_results = [
        result
        for result in results
        if result["answerable"]
    ]

    successful_retrievals = [
        result
        for result in answerable_results
        if result["retrieval_hit"]
    ]

    retrieval_at_k = (
        len(successful_retrievals)
        / len(answerable_results)
    )

    print("\n--------------------------------")
    print(f"RETRIEVAL@{TOP_K}")
    print("--------------------------------")

    print(
        f"{len(successful_retrievals)} / "
        f"{len(answerable_results)} "
        f"answerable questions retrieved "
        f"an expected source."
    )

    print(f"Retrieval@{TOP_K}: {retrieval_at_k:.3f}")

    # -----------------------------
    # Print question-level results
    # -----------------------------

    print("\n--------------------------------")
    print("QUESTION-LEVEL RESULTS")
    print("--------------------------------")

    for result in results:

        print(f"\n{result['id']}")
        print(f"Question: {result['question']}")

        print(
            f"Answerable: "
            f"{result['answerable']}"
        )

        print(
            f"Expected sources: "
            f"{result['expected_sources']}"
        )

        print(
            f"Retrieved sources: "
            f"{result['retrieved_sources']}"
        )

        if result["answerable"]:
            print(
                f"Retrieval hit: "
                f"{result['retrieval_hit']}"
            )

        print("Top chunks:")

        for rank, retrieved in enumerate(
            result["top_results"],
            start=1
        ):

            print(
                f"  {rank}. "
                f"{retrieved['filename']} "
                f"(p. {retrieved['page_number']}) "
                f"score={retrieved['similarity']:.4f}"
            )

    # -----------------------------
    # Save results
    # -----------------------------

    with open(results_file, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n--------------------------------")
    print("RESULTS SAVED")
    print("--------------------------------")

    print(f"File: {results_file}")