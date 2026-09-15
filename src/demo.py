from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from retriever import (
    load_chunks,
    load_embeddings,
    retrieve
)

from evidence_verifier import verify_evidence

from generator import (
    generate_answer,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    TOP_K
)


# --------------------------------
# PATHS
# --------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

CHUNKS_FILE = (
    PROCESSED_DIR / "chunks.json"
)

EMBEDDINGS_FILE = (
    PROCESSED_DIR / "embeddings.npy"
)


# --------------------------------
# LOAD DATA
# --------------------------------

print("\n================================")
print("EVIDENCERAG INTERACTIVE DEMO")
print("================================")

print("\n--------------------------------")
print("LOADING DATA")
print("--------------------------------")

chunks = load_chunks(
    CHUNKS_FILE
)

embeddings = load_embeddings(
    EMBEDDINGS_FILE
)

print(
    f"Chunks: {len(chunks)}"
)

print(
    f"Embeddings: {embeddings.shape}"
)


# --------------------------------
# LOAD MODELS
# --------------------------------

print("\n--------------------------------")
print("LOADING MODELS")
print("--------------------------------")

print(
    f"Embedding model: "
    f"{EMBEDDING_MODEL_NAME}"
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print(
    f"Generation model: "
    f"{GENERATION_MODEL_NAME}"
)

tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    GENERATION_MODEL_NAME
)


# --------------------------------
# READY
# --------------------------------

print("\n================================")
print("READY")
print("================================")

print(
    "\nAsk a question about the "
    "research corpus."
)

print(
    "Type 'exit' to stop the demo.\n"
)


# --------------------------------
# INTERACTIVE LOOP
# --------------------------------

while True:

    question = input(
        "Question: "
    ).strip()


    # --------------------------------
    # EXIT
    # --------------------------------

    if question.lower() in {
        "exit",
        "quit",
        "q"
    }:

        print(
            "\nExiting EvidenceRAG demo."
        )

        break


    if not question:

        print(
            "Please enter a question.\n"
        )

        continue


    # --------------------------------
    # RETRIEVE
    # --------------------------------

    retrieved_chunks = retrieve(
        question,
        embedding_model,
        chunks,
        embeddings,
        top_k=TOP_K
    )

    strongest_similarity = max(
        chunk["similarity"]
        for chunk in retrieved_chunks
    )


    # --------------------------------
    # VERIFY EVIDENCE
    # --------------------------------

    decision = verify_evidence(
        question,
        retrieved_chunks,
        tokenizer,
        model
    )


    print(
        f"\nStrongest similarity: "
        f"{strongest_similarity:.4f}"
    )

    print(
        f"Evidence decision: "
        f"{decision}"
    )


    # --------------------------------
    # ANSWER / REFUSE
    # --------------------------------

    if decision == "SUPPORTED":

        answer = generate_answer(
            question,
            retrieved_chunks,
            tokenizer,
            model
        )

        status = "ANSWERED"


    elif decision == "CONFLICTING":

        answer = generate_answer(
            question,
            retrieved_chunks,
            tokenizer,
            model
        )

        status = "ANSWERED_WITH_CONFLICT"


    else:

        answer = (
            "I do not have enough evidence "
            "in the provided research corpus "
            "to answer this question reliably."
        )

        status = "REFUSED"


    # --------------------------------
    # DISPLAY RESULT
    # --------------------------------

    print(
        f"Status: {status}"
    )

    print(
        f"Answer: {answer}"
    )

    print(
        "\n" + "-" * 60 + "\n"
    )