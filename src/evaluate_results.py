from pathlib import Path
import json


# --------------------------------
# LOAD RESULTS
# --------------------------------

def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# --------------------------------
# EVALUATION
# --------------------------------

def evaluate_results(
    questions,
    basic_results,
    evidence_results
):

    # Ground-truth answerability
    answerable_questions = [
        q for q in questions
        if q["answerable"]
    ]

    unanswerable_questions = [
        q for q in questions
        if not q["answerable"]
    ]


    # Create lookup dictionaries
    basic_by_id = {
        result["id"]: result
        for result in basic_results
    }

    evidence_by_id = {
        result["id"]: result
        for result in evidence_results
    }


    # --------------------------------
    # BASIC RAG
    # --------------------------------

    basic_unanswerable_answers = 0

    for q in basic_results:

        if not q["answerable"]:

            answer = q["answer"].strip()

            if answer:
                basic_unanswerable_answers += 1


    # --------------------------------
    # EVIDENCERAG
    # --------------------------------

    evidence_answered = 0
    evidence_refused_unanswerable = 0
    evidence_false_refusals = 0
    evidence_conflicting = 0

    for q in questions:

        result = evidence_by_id[q["id"]]

        status = result["status"]


        # --------------------------------
        # ANSWERABLE QUESTIONS
        # --------------------------------

        if q["answerable"]:

            if status in [
                "ANSWERED",
                "ANSWERED_WITH_CONFLICT"
            ]:

                evidence_answered += 1

            elif status == "REFUSED":

                evidence_false_refusals += 1


        # --------------------------------
        # UNANSWERABLE QUESTIONS
        # --------------------------------

        else:

            if status == "REFUSED":

                evidence_refused_unanswerable += 1


        # --------------------------------
        # CONFLICTING CASES
        # --------------------------------

        if status == "ANSWERED_WITH_CONFLICT":

            evidence_conflicting += 1


    # --------------------------------
    # METRICS
    # --------------------------------

    total_answerable = len(
        answerable_questions
    )

    total_unanswerable = len(
        unanswerable_questions
    )


    supported_coverage = (
        evidence_answered
        / total_answerable
        * 100
    )


    correct_refusal_rate = (
        evidence_refused_unanswerable
        / total_unanswerable
        * 100
    )


    false_refusal_rate = (
        evidence_false_refusals
        / total_answerable
        * 100
    )


    # --------------------------------
    # BASIC RAG UNSUPPORTED-ANSWER RATE
    # --------------------------------

    basic_unsupported_rate = (
        basic_unanswerable_answers
        / total_unanswerable
        * 100
    )


    # --------------------------------
    # EVIDENCERAG UNSUPPORTED-ANSWER RATE
    # --------------------------------

    evidence_unsupported_answers = (
        total_unanswerable
        - evidence_refused_unanswerable
    )


    evidence_unsupported_rate = (
        evidence_unsupported_answers
        / total_unanswerable
        * 100
    )


    # --------------------------------
    # PRINT METRICS
    # --------------------------------

    print("\n================================")
    print("EVIDENCERAG EVALUATION")
    print("================================")


    print("\nDataset")
    print("--------------------------------")

    print(
        f"Total questions: "
        f"{len(questions)}"
    )

    print(
        f"Answerable questions: "
        f"{total_answerable}"
    )

    print(
        f"Unanswerable questions: "
        f"{total_unanswerable}"
    )


    print("\nBasic RAG")
    print("--------------------------------")

    print(
        f"Unsupported answers on "
        f"unanswerable questions: "
        f"{basic_unanswerable_answers}/"
        f"{total_unanswerable}"
    )

    print(
        f"Unsupported-answer rate: "
        f"{basic_unsupported_rate:.1f}%"
    )


    print("\nEvidenceRAG")
    print("--------------------------------")

    print(
        f"Answerable questions allowed: "
        f"{evidence_answered}/"
        f"{total_answerable}"
    )

    print(
        f"Supported-question coverage: "
        f"{supported_coverage:.1f}%"
    )

    print(
        f"Unanswerable questions correctly refused: "
        f"{evidence_refused_unanswerable}/"
        f"{total_unanswerable}"
    )

    print(
        f"Correct refusal rate: "
        f"{correct_refusal_rate:.1f}%"
    )

    print(
        f"False refusals on answerable questions: "
        f"{evidence_false_refusals}/"
        f"{total_answerable}"
    )

    print(
        f"False-refusal rate: "
        f"{false_refusal_rate:.1f}%"
    )

    print(
        f"Conflicting cases detected: "
        f"{evidence_conflicting}"
    )

    print(
        f"Unsupported answers on unanswerable "
        f"questions: "
        f"{evidence_unsupported_answers}/"
        f"{total_unanswerable}"
    )

    print(
        f"Unsupported-answer rate: "
        f"{evidence_unsupported_rate:.1f}%"
    )


    # --------------------------------
    # ANSWER COMPARISON
    # --------------------------------

    print("\n================================")
    print("BASIC RAG vs EVIDENCERAG ANSWERS")
    print("================================")


    for q in questions:

        question_id = q["id"]

        basic_result = basic_by_id[
            question_id
        ]

        evidence_result = evidence_by_id[
            question_id
        ]


        print("\n--------------------------------")
        print(question_id)
        print("--------------------------------")

        print(
            f"Question:\n"
            f"{q['question']}"
        )

        print(
            f"\nQuestion type: "
            f"{q['type']}"
        )

        print(
            f"Ground truth answerable: "
            f"{q['answerable']}"
        )


        # --------------------------------
        # BASIC RAG ANSWER
        # --------------------------------

        print("\nBasic RAG answer:")

        print(
            basic_result["answer"]
        )


        # --------------------------------
        # EVIDENCERAG RESULT
        # --------------------------------

        print(
            "\nEvidenceRAG status:"
        )

        print(
            evidence_result["status"]
        )


        print(
            "\nEvidenceRAG answer:"
        )

        print(
            evidence_result["answer"]
        )


        # --------------------------------
        # SIMILARITY
        # --------------------------------

        if "strongest_similarity" in evidence_result:

            print(
                "\nStrongest similarity:"
            )

            print(
                f"{evidence_result['strongest_similarity']:.4f}"
            )


    # --------------------------------
    # SHORT SUMMARY
    # --------------------------------

    print("\n================================")
    print("EXPERIMENT SUMMARY")
    print("================================")

    print(
        "\nBasic RAG:"
    )

    print(
        f"- Unsupported-answer rate: "
        f"{basic_unsupported_rate:.1f}%"
    )


    print(
        "\nEvidenceRAG:"
    )

    print(
        f"- Supported-question coverage: "
        f"{supported_coverage:.1f}%"
    )

    print(
        f"- Correct refusal rate: "
        f"{correct_refusal_rate:.1f}%"
    )

    print(
        f"- False-refusal rate: "
        f"{false_refusal_rate:.1f}%"
    )

    print(
        f"- Unsupported-answer rate: "
        f"{evidence_unsupported_rate:.1f}%"
    )


# --------------------------------
# MAIN
# --------------------------------

if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    experiments_dir = (
        project_root
        / "experiments"
    )


    questions_file = (
        experiments_dir
        / "questions.json"
    )

    basic_file = (
        experiments_dir
        / "basic_rag_results.json"
    )

    evidence_file = (
        experiments_dir
        / "evidencerag_final_results.json"
    )


    # --------------------------------
    # LOAD FILES
    # --------------------------------

    print("--------------------------------")
    print("LOADING EXPERIMENT RESULTS")
    print("--------------------------------")


    questions = load_json(
        questions_file
    )

    basic_results = load_json(
        basic_file
    )

    evidence_results = load_json(
        evidence_file
    )


    print(
        f"Questions loaded: "
        f"{len(questions)}"
    )

    print(
        f"Basic RAG results: "
        f"{len(basic_results)}"
    )

    print(
        f"EvidenceRAG results: "
        f"{len(evidence_results)}"
    )


    # --------------------------------
    # RUN EVALUATION
    # --------------------------------

    evaluate_results(
        questions,
        basic_results,
        evidence_results
    )