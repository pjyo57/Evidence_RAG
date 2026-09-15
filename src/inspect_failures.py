from pathlib import Path
import json


# --------------------------------
# LOAD RESULTS
# --------------------------------

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
# INSPECT Q10 AND Q11
# --------------------------------

for result in results:

    if result["id"] not in ["Q10", "Q11"]:
        continue

    print("\n")
    print("=" * 80)
    print(result["id"])
    print("=" * 80)

    print("\nQuestion:")
    print(result["question"])

    print(
        f"\nAnswerable according to our dataset: "
        f"{result['answerable']}"
    )

    print(
        f"Strongest similarity: "
        f"{result['strongest_similarity']:.4f}"
    )

    print("\nRetrieved evidence:")

    for i, chunk in enumerate(
        result["retrieved_chunks"],
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"Rank: {i}"
        )

        print(
            f"File: {chunk['filename']}"
        )

        print(
            f"Page: {chunk['page_number']}"
        )

        print(
            f"Similarity: "
            f"{chunk['similarity']:.4f}"
        )

        print("\nText:")
        print(
            chunk["text"][:1200]
        )


print("\n")
print("=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)