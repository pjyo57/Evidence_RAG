from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------
# PARAMETERS
# --------------------------------

VERIFICATION_MODEL_NAME = "google/flan-t5-base"

MAX_VERIFICATION_TOKENS = 20


# --------------------------------
# BUILD VERIFICATION PROMPT
# --------------------------------

def build_verification_prompt(
    question,
    retrieved_chunks
):
    """
    Build a strict evidence-verification prompt.

    The verifier must distinguish:
    - direct evidence
    - merely related information
    - conflicting evidence
    """

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
You are an evidence verifier.

Your task is to determine whether the provided evidence
actually supports an answer to the question.

Use ONLY the provided evidence.

Definitions:

SUPPORTED:
The evidence directly contains information needed
to answer the question without requiring unsupported
assumptions.

INSUFFICIENT:
The evidence is only related to the topic, discusses
similar concepts, or does not contain enough information
to answer the question.

CONFLICTING:
The evidence contains different or incompatible claims
that cannot be resolved from the provided evidence.

Important:
Do NOT classify evidence as SUPPORTED merely because
it discusses the same topic as the question.

A very strong or universal claim must have direct evidence
supporting that claim.

For example, evidence discussing several methods does NOT
support the claim that one method is definitively the best
for every model and every task.

Return exactly one label:

SUPPORTED

INSUFFICIENT

or

CONFLICTING


Question:
{question}

Evidence:
{evidence_text}

Label:
"""

    return prompt


# --------------------------------
# VERIFY EVIDENCE
# --------------------------------

def verify_evidence(
    question,
    retrieved_chunks,
    tokenizer,
    model
):
    """
    Classify the retrieved evidence.
    """

    prompt = build_verification_prompt(
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
        max_new_tokens=MAX_VERIFICATION_TOKENS,
        do_sample=False
    )

    decision = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip().upper()


    # --------------------------------
    # CONTROL OUTPUT
    # --------------------------------

    if decision == "SUPPORTED":

        return "SUPPORTED"

    elif decision == "CONFLICTING":

        return "CONFLICTING"

    else:

        return "INSUFFICIENT"


# --------------------------------
# CONTROLLED TEST
# --------------------------------

if __name__ == "__main__":

    from pathlib import Path
    import json


    # --------------------------------
    # FIND RESULTS FILE
    # --------------------------------

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    results_file = (
        project_root
        / "experiments"
        / "evidencerag_results.json"
    )


    # --------------------------------
    # LOAD RESULTS
    # --------------------------------

    with open(
        results_file,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)


    # --------------------------------
    # LOAD MODEL
    # --------------------------------

    print("--------------------------------")
    print("LOADING VERIFICATION MODEL")
    print("--------------------------------")

    tokenizer = AutoTokenizer.from_pretrained(
        VERIFICATION_MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        VERIFICATION_MODEL_NAME
    )


    # --------------------------------
    # TEST QUESTIONS
    # --------------------------------

    test_ids = [
      "Q1",
      "Q2",
      "Q3",
      "Q4",
      "Q5",
      "Q6",
      "Q7",
      "Q8",
      "Q9",
      "Q10",
      "Q11",
      "Q12"
    ]

    print("\n================================")
    print("STRICT EVIDENCE VERIFICATION")
    print("================================")


    for result in results:

        if result["id"] not in test_ids:
            continue


        question = result["question"]

        retrieved_chunks = (
            result["retrieved_chunks"]
        )


        print("\n--------------------------------")
        print(result["id"])
        print("--------------------------------")

        print(
            f"Question: {question}"
        )

        print(
            f"Dataset answerable: "
            f"{result['answerable']}"
        )

        print(
            f"Strongest similarity: "
            f"{result['strongest_similarity']:.4f}"
        )


        decision = verify_evidence(
            question,
            retrieved_chunks,
            tokenizer,
            model
        )


        print(
            f"Verifier decision: "
            f"{decision}"
        )