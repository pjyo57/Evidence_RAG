# -----------------------------
# Evidence gate parameters
# -----------------------------

EVIDENCE_THRESHOLD = 0.50


def check_evidence(retrieved_chunks, threshold=EVIDENCE_THRESHOLD):
    """
    Decide whether the retrieved evidence is strong enough
    to allow answer generation.

    The gate currently uses the highest similarity score
    among the retrieved chunks.
    """

    if not retrieved_chunks:
        return {
            "sufficient": False,
            "score": 0.0
        }

    strongest_score = max(
        chunk["similarity"]
        for chunk in retrieved_chunks
    )

    sufficient = strongest_score >= threshold

    return {
        "sufficient": sufficient,
        "score": strongest_score
    }


if __name__ == "__main__":

    # Small demonstration
    example_chunks = [
        {"similarity": 0.42},
        {"similarity": 0.55},
        {"similarity": 0.48}
    ]

    result = check_evidence(
        example_chunks
    )

    print("--------------------------------")
    print("EVIDENCE GATE TEST")
    print("--------------------------------")

    print(
        f"Strongest similarity: "
        f"{result['score']:.4f}"
    )

    print(
        f"Threshold: "
        f"{EVIDENCE_THRESHOLD:.4f}"
    )

    print(
        f"Evidence sufficient: "
        f"{result['sufficient']}"
    )