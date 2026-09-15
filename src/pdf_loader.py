from pathlib import Path
from pypdf import PdfReader


def extract_pages_from_pdf(pdf_path):
    """Extract text from a PDF while preserving page numbers."""

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:
            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


def load_papers(data_dir):
    """Load all PDF papers and preserve page-level text."""

    data_dir = Path(data_dir)

    papers = []

    pdf_files = sorted(data_dir.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:

        print(f"Reading: {pdf_path.name}")

        pages = extract_pages_from_pdf(pdf_path)

        paper = {
            "doc_id": pdf_path.stem,
            "filename": pdf_path.name,
            "pages": pages
        }

        papers.append(paper)

    return papers


if __name__ == "__main__":

    # Find the project root automatically
    project_root = Path(__file__).resolve().parent.parent

    papers_dir = project_root / "data" / "raw" / "papers"

    papers = load_papers(papers_dir)

    print("\n--------------------------------")
    print("PDF LOADING COMPLETE")
    print("--------------------------------")

    print(f"Number of papers: {len(papers)}")

    for paper in papers:

        word_count = sum(
            len(page["text"].split())
            for page in paper["pages"]
        )

        print(
            f"{paper['filename']}: "
            f"{len(paper['pages'])} pages, "
            f"{word_count:,} words"
        )

    # Inspect the first paper
    print("\n--------------------------------")
    print("PAGE-LEVEL TEXT INSPECTION")
    print("--------------------------------")

    first_paper = papers[0]

    print(f"File: {first_paper['filename']}")
    print(f"Document ID: {first_paper['doc_id']}")

    first_page = first_paper["pages"][0]

    print(f"Page number: {first_page['page_number']}")

    print("\nFirst page text:\n")
    print(first_page["text"][:2000])