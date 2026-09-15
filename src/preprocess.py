from pathlib import Path

from pdf_loader import load_papers
from cleaner import clean_text


def preprocess_papers(papers_dir):
    """
    Load all PDFs and clean their page-level text.
    """

    papers = load_papers(papers_dir)

    for paper in papers:

        for page in paper["pages"]:

            page["text"] = clean_text(page["text"])

    return papers


if __name__ == "__main__":

    # Find project root automatically
    project_root = Path(__file__).resolve().parent.parent

    papers_dir = project_root / "data" / "raw" / "papers"

    papers = preprocess_papers(papers_dir)

    print("\n--------------------------------")
    print("PREPROCESSING COMPLETE")
    print("--------------------------------")

    print(f"Number of papers: {len(papers)}")

    total_pages = 0
    total_words = 0

    for paper in papers:

        pages = paper["pages"]

        word_count = sum(
            len(page["text"].split())
            for page in pages
        )

        total_pages += len(pages)
        total_words += word_count

        print(
            f"{paper['filename']}: "
            f"{len(pages)} pages, "
            f"{word_count:,} cleaned words"
        )

    print("\n--------------------------------")
    print("CORPUS SUMMARY")
    print("--------------------------------")

    print(f"Total papers: {len(papers)}")
    print(f"Total pages: {total_pages}")
    print(f"Total cleaned words: {total_words:,}")

    # Inspect one real page
    print("\n--------------------------------")
    print("REAL PAPER INSPECTION")
    print("--------------------------------")

    first_paper = papers[0]
    first_page = first_paper["pages"][0]

    print(f"File: {first_paper['filename']}")
    print(f"Page: {first_page['page_number']}")

    print("\nCleaned text:\n")
    print(first_page["text"][:2000])