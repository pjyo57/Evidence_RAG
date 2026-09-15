from pathlib import Path
import json


# --------------------------------
# PARAMETERS
# --------------------------------

THRESHOLDS = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75
]


# --------------------------------
# LOAD RESULTS
# --------------------------------

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    results_file = (
        project_root
        / "experiments"
        / "evidencerag_results.json"
    )

    with open(
        results_file,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)


    # --------------------------------
    # THRESHOLD EXPERIMENT
    # --------------------------------

    print("================================")
    print("EVIDENCE THRESHOLD EXPERIMENT")
    print("================================")

    print(
        "\nWe test how the evidence gate behaves "
        "at different similarity thresholds."
    )

    print("\n")


    for threshold in THRESHOLDS:

        answerable_questions = [
            r for r in results
            if r["answerable"] is True
        ]

        unanswerable_questions = [
            r for r in results
            if r["answerable"] is False
        ]


        # Questions allowed through gate
        answerable_passed = sum(
            r["strongest_similarity"] >= threshold
            for r in answerable_questions
        )

        unanswerable_blocked = sum(
            r["strongest_similarity"] < threshold
            for r in unanswerable_questions
        )


        answerable_coverage = (
            answerable_passed
            / len(answerable_questions)
        )

        unanswerable_block_rate = (
            unanswerable_blocked
            / len(unanswerable_questions)
        )


        print("--------------------------------")
        print(
            f"Threshold: {threshold:.2f}"
        )

        print(
            f"Answerable questions allowed: "
            f"{answerable_passed}/"
            f"{len(answerable_questions)} "
            f"({answerable_coverage:.1%})"
        )

        print(
            f"Unanswerable questions blocked: "
            f"{unanswerable_blocked}/"
            f"{len(unanswerable_questions)} "
            f"({unanswerable_block_rate:.1%})"
        )


    # --------------------------------
    # QUESTION SCORES
    # --------------------------------

    print("\n================================")
    print("QUESTION SIMILARITY SCORES")
    print("================================")

    for result in results:

        print(
            f"{result['id']}: "
            f"{result['strongest_similarity']:.4f} "
            f"| answerable = "
            f"{result['answerable']}"
        )