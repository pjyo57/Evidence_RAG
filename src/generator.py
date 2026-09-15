
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


# --------------------------------
# PARAMETERS
# --------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-base"

TOP_K = 5
MAX_NEW_TOKENS = 150

# Evidence verifier parameters
VERIFICATION_MAX_TOKENS = 20


# --------------------------------
# LOAD QUESTIONS
# --------------------------------

def load_questions(questions_file):

    with open(
        questions_file,
        "r",
        encoding="utf-8"
    ) as f:

        questions = json.load(f)

    return questions


# --------------------------------
# BUILD ANSWER PROMPT
# --------------------------------

def build_prompt(question, retrieved_chunks):

    evidence_text = ""

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        evidence_text += (
            f"\nEvidence {i} "
            f"({chunk['filename']}, "
            f"page {chunk['page_number']}):\n"
        )

        evidence_text += chunk["text"]
        evidence_text += "\n"

    prompt = f"""
Answer the question using the evidence provided below.

Use the evidence as the primary source for your answer.
Do not invent facts, numbers, names, or results.

Question:
{question}

Evidence:
{evidence_text}

Answer:
"""

    return prompt


# --------------------------------
# GENERATE ANSWER
# --------------------------------

def generate_answer(
    question,
    retrieved_chunks,
    tokenizer,
    model
):

    prompt = build_prompt(
        question,
        retrieved_chunks
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()


# --------------------------------
# MAIN
# --------------------------------

if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    processed_dir = (
        project_root / "data" / "processed"
    )

    experiments_dir = (
        project_root / "experiments"
    )

    chunks_file = (
        processed_dir / "chunks.json"
    )

    embeddings_file = (
        processed_dir / "embeddings.npy"
    )

    questions_file = (
        experiments_dir / "questions.json"
    )

    output_file = (
        experiments_dir /
        "evidencerag_final_results.json"
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

    questions = load_questions(
        questions_file
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Embeddings: {embeddings.shape}"
    )

    print(
        f"Questions: {len(questions)}"
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
    # RUN EVIDENCERAG
    # --------------------------------

    results = []

    print("\n================================")
    print("RUNNING FINAL EVIDENCERAG")
    print("================================")


    for question_data in questions:

        question_id = question_data["id"]

        question = question_data["question"]

        print("\n--------------------------------")
        print(f"{question_id}")
        print("--------------------------------")

        print(
            f"Question: {question}"
        )


        # --------------------------------
        # RETRIEVE EVIDENCE
        # --------------------------------

        retrieved_chunks = retrieve(
            question,
            embedding_model,
            chunks,
            embeddings,
            top_k=TOP_K
        )

        strongest_score = max(
            chunk["similarity"]
            for chunk in retrieved_chunks
        )


        print(
            f"Strongest similarity: "
            f"{strongest_score:.4f}"
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
            f"Evidence decision: "
            f"{decision}"
        )


        # --------------------------------
        # EVIDENCE-AWARE DECISION
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


        print(
            f"Status: {status}"
        )

        print(
            f"Answer: {answer}"
        )


        # --------------------------------
        # SAVE RESULT
        # --------------------------------

        results.append({

            "id": question_id,

            "question": question,

            "question_type":
                question_data["type"],

            "answerable":
                question_data["answerable"],

            "expected_sources":
                question_data["expected_sources"],

            "top_k":
                TOP_K,

            "strongest_similarity":
                strongest_score,

            "verification_decision":
                decision,

            "status":
                status,

            "answer":
                answer,

            "retrieved_chunks":
                retrieved_chunks
        })


    # --------------------------------
    # SAVE RESULTS
    # --------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )


    print("\n================================")
    print("FINAL EVIDENCERAG COMPLETE")
    print("================================")

    print(
        f"Results saved to: "
        f"{output_file}"
    )
