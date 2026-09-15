import re


def clean_text(text):
    """
    Clean extracted PDF text while preserving meaningful scientific information.
    """

    # 1. Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 2. Join words split across a line by PDF hyphenation
    # Example:
    # "factual con-\n"
    # "sistency"
    # becomes:
    # "factual consistency"
    text = re.sub(r"-\s*\n\s*(?=\w)", "", text)

    # 3. Replace remaining line breaks with spaces
    text = text.replace("\n", " ")

    # 4. Collapse multiple spaces into a single space
    text = re.sub(r"\s+", " ", text)

    # 5. Remove leading/trailing whitespace
    text = text.strip()

    return text


if __name__ == "__main__":

    sample_text = """
    This is an example of extracted
    PDF text with     extra spaces.

    Factual con-
    sistency is important.

    This is a state-of-the-art model.
    """

    print("--------------------------------")
    print("ORIGINAL TEXT")
    print("--------------------------------")
    print(sample_text)

    cleaned = clean_text(sample_text)

    print("\n--------------------------------")
    print("CLEANED TEXT")
    print("--------------------------------")
    print(cleaned)