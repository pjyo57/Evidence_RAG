from pathlib import Path
import json
import re


# --------------------------------
# PARAMETERS
# --------------------------------

REFERENCE_PATTERNS = [
    r"\bReferences\b",
    r"\bBibliography\b",
    r"\barXiv\b",
    r"\bProceedings\b",
    r"\bdoi\.org\b",
    r"\bet al\.",
]


# --------------------------------
# REFERENCE DETECTION
# --------------------------------
def looks_like_reference_chunk(text):
    """
    Conservative heuristic for identifying bibliography-like
    chunks.

    We require multiple citation-style entries rather than
    merely detecting words such as 'arXiv' or 'et al.'.
    """

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

    # Require several indicators.
    return matches >= 3


# --------------------------------
# MAIN
# --------------------------------

if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    chunks_file = (
        project_root
        / "data"
        / "processed"
        / "chunks.json"
    )

    results_file = (
      project_root
      / "experiments"
      / "evidencerag_results.json"
   )


    # --------------------------------
    # LOAD CHUNKS
    # --------------------------------

    with open(
        chunks_file,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)


    # --------------------------------
    # ANALYZE ALL CHUNKS
    # --------------------------------

    reference_chunks = []

    for chunk in chunks:

        if looks_like_reference_chunk(
            chunk["text"]
        ):

            reference_chunks.append(
                chunk
            )


    print("================================")
    print("REFERENCE CHUNK ANALYSIS")
    print("================================")

    print(
        f"Total chunks: {len(chunks)}"
    )

    print(
        f"Reference-like chunks: "
        f"{len(reference_chunks)}"
    )

    percentage = (
        len(reference_chunks)
        / len(chunks)
        * 100
    )

    print(
        f"Percentage: {percentage:.2f}%"
    )


    # --------------------------------
    # INSPECT EXAMPLES
    # --------------------------------

    print("\n================================")
    print("REFERENCE-LIKE CHUNK EXAMPLES")
    print("================================")


    for i, chunk in enumerate(
        reference_chunks[:10],
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"Example {i}"
        )

        print(
            f"File: {chunk['filename']}"
        )

        print(
            f"Page: {chunk['page_number']}"
        )

        print(
            f"Chunk: {chunk['chunk_number']}"
        )

        print("\nText:")

        print(
            chunk["text"][:1000]
        )


    # --------------------------------
    # ANALYZE RETRIEVAL RESULTS
    # --------------------------------

    if results_file.exists():

        with open(
            results_file,
            "r",
            encoding="utf-8"
        ) as f:

            retrieval_results = json.load(f)


        print("\n================================")
        print("REFERENCE CHUNKS IN RETRIEVAL")
        print("================================")


        total_retrieved = 0
        reference_retrieved = 0


        for result in retrieval_results:

            retrieved_chunks = (
                result.get(
                    "retrieved_chunks",
                    []
                )
            )


            for chunk in retrieved_chunks:

                total_retrieved += 1

                if looks_like_reference_chunk(
                    chunk["text"]
                ):

                    reference_retrieved += 1


        print(
            f"Total retrieved chunks: "
            f"{total_retrieved}"
        )

        print(
            f"Reference-like retrieved chunks: "
            f"{reference_retrieved}"
        )


        if total_retrieved > 0:

            retrieved_percentage = (
                reference_retrieved
                / total_retrieved
                * 100
            )

            print(
                f"Percentage of retrieved chunks "
                f"that are reference-like: "
                f"{retrieved_percentage:.2f}%"
            )


    else:

        print(
            "\nretrieval_results.json was not found."
        )

        print(
            "Only the full chunk analysis was performed."
        )


    print("\n================================")
    print("ANALYSIS COMPLETE")
    print("================================")