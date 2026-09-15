from pathlib import Path
import json

from preprocess import preprocess_papers


# -----------------------------
# Chunking parameters
# -----------------------------

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def create_chunks(papers, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split cleaned page-level text into overlapping word-based chunks.

    Each chunk keeps:
    - document ID
    - filename
    - page number
    - chunk ID
    - chunk text
    """

    chunks = []

    for paper in papers:

        for page in paper["pages"]:

            words = page["text"].split()

            start = 0
            chunk_number = 0

            while start < len(words):

                end = start + chunk_size

                chunk_words = words[start:end]

                if not chunk_words:
                    break

                chunk = {
                    "chunk_id": (
                        f"{paper['doc_id']}_"
                        f"p{page['page_number']}_"
                        f"c{chunk_number}"
                    ),
                    "doc_id": paper["doc_id"],
                    "filename": paper["filename"],
                    "page_number": page["page_number"],
                    "chunk_number": chunk_number,
                    "text": " ".join(chunk_words)
                }

                chunks.append(chunk)

                chunk_number += 1

                # Move forward while keeping overlap
                start += chunk_size - chunk_overlap

    return chunks


if __name__ == "__main__":

    # Find project root automatically
    project_root = Path(__file__).resolve().parent.parent

    papers_dir = project_root / "data" / "raw" / "papers"
    processed_dir = project_root / "data" / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean all papers
    papers = preprocess_papers(papers_dir)

    # Create chunks
    chunks = create_chunks(papers)

    # Save chunks
    output_file = processed_dir / "chunks.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n--------------------------------")
    print("CHUNKING COMPLETE")
    print("--------------------------------")

    print(f"Number of papers: {len(papers)}")
    print(f"Chunk size: {CHUNK_SIZE} words")
    print(f"Chunk overlap: {CHUNK_OVERLAP} words")
    print(f"Total chunks: {len(chunks)}")

    print("\n--------------------------------")
    print("SAMPLE CHUNKS")
    print("--------------------------------")

    for chunk in chunks[:3]:

        print("\nChunk ID:", chunk["chunk_id"])
        print("File:", chunk["filename"])
        print("Page:", chunk["page_number"])
        print("Words:", len(chunk["text"].split()))

        print("\nText:")
        print(chunk["text"][:1000])
        print("\n" + "-" * 60)