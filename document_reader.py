import hashlib
import re
from pathlib import Path

import pymupdf  # PyMuPDF

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

# Bump this when the extraction/chunking code changes, so existing
# indexes are rebuilt automatically.
INDEX_VERSION = "5"


# Words that end a wrapped line of ordinary text, never a heading.
_NOT_HEADING_WORDS = {
    "hour", "hours", "min", "mins", "minute", "minutes", "day", "days",
    "month", "months", "year", "years", "the", "of", "and", "or", "to",
    "a", "an", "in", "on", "at", "for", "by", "with", "is", "are", "be",
}


def _ends_sentence(line):
    return line.rstrip().endswith((".", ":", ";", "?", "!", ")"))


def looks_like_heading(line, previous=None):
    """
    Short lines such as 'Overtime', 'OVERTIME RULES' or '3.2 Hotel Stay'.

    `previous` is the line above it. A heading starts a new paragraph, so
    one or two words on their own right after an unfinished line ("...at" /
    "Karachi") are just the end of a wrapped sentence.
    """
    line = line.strip()

    # "TRANSPORT TEAM (for the period to 31 Dec 2026)": judge the title,
    # not the note in brackets (closed, or cut off by a line break).
    title = re.sub(r"\s*\([^)]*\)?\s*$", "", line).strip()
    has_note = title != line
    line = title

    if len(line) > 50 or len(line.split()) > 6:
        return False

    if not re.search(r"[A-Za-z]", line):
        return False

    if line.endswith((".", ",", ";", ":")) or ":" in line:
        return False

    # "3. Overtime" or "3.2 Hotel Stay"
    if re.match(r"^\d+(\.\d+)*[\).]\s+[A-Za-z]", line) or re.match(
        r"^\d+(\.\d+)+\s+[A-Za-z]", line
    ):
        return True

    # Plain headings have no digits; lines like "1 2 3 Rs.6,000" are data.
    if re.search(r"\d", line):
        return False

    if not (line.isupper() or line.istitle()):
        return False

    words = line.split()

    # (A wrapped word is plain text; ALL CAPS, or a title with a bracketed
    # note, is a heading even when the paragraph above has no full stop.)
    if len(words) <= 2 and previous is not None and not (
        has_note or line.isupper()
    ):
        if words[-1].lower() in _NOT_HEADING_WORDS:
            return False

        # After an unfinished line (not a sentence end, table row or
        # another heading) this is a wrapped word, not a heading.
        prev = previous.strip()

        if prev:
            # Some PDFs put a list number at the END of each item ("... at 1."),
            # which is not a sentence end: look at the words before it.
            bare = re.sub(r"\s+\d{1,2}[.)]$", "", prev)

            if bare != prev:
                last = bare.split()[-1].lower().strip(",;") if bare.split() else ""
                finished = last not in _NOT_HEADING_WORDS
            else:
                finished = (
                    _ends_sentence(prev)
                    or bool(re.search(r"\d$", prev))
                    or looks_like_heading(prev)
                )

            if not finished:
                return False

    return True


def looks_like_soft_heading(line, previous, following):
    """
    A heading written in ordinary sentence case ("Privilege leave"): short,
    capitalised, no punctuation, between a finished sentence and a new one.
    """
    line = line.strip()
    words = line.split()

    if not (2 <= len(words) <= 5) or len(line) > 40:
        return False

    if re.search(r"\d", line) or line.endswith((".", ",", ";", ":", "?")):
        return False

    if not line[0].isupper() or words[-1].lower() in _NOT_HEADING_WORDS:
        return False

    if previous is None or not previous.strip().endswith((".", "?", "!")):
        return False

    return bool(following) and following.strip()[:1].isupper()


def pdf_signature():
    """
    Fingerprint of the PDFs in the documents folder (names, sizes, times).
    When it changes, the search indexes are out of date.
    """
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(INDEX_VERSION.encode())

    for pdf_path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        stat = pdf_path.stat()
        digest.update(
            f"{pdf_path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode()
        )

    return digest.hexdigest()


def ocr_page_text(page):
    """
    Try OCR for a scanned page. Needs Tesseract installed on the machine;
    returns an empty string if it isn't available.
    """
    try:
        textpage = page.get_textpage_ocr(dpi=200, full=True)
        return page.get_text("text", textpage=textpage).strip()
    except Exception:
        return ""


def extract_pdf_text(pdf_path):
    """Extract page text (line breaks kept) while preserving page numbers."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True).strip()

            if not text:
                text = ocr_page_text(page)

                if text:
                    print(f"OCR used: {pdf_path.name}, page {page_number}")
                else:
                    print(
                        f"Warning: no text found in {pdf_path.name}, "
                        f"page {page_number} (scanned page? install "
                        f"Tesseract to enable OCR)."
                    )

            if text:
                pages.append({
                    "document": pdf_path.name,
                    "page": page_number,
                    "text": text
                })

    return pages


def extract_all_pdfs():
    all_pages = []

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        try:
            all_pages.extend(extract_pdf_text(pdf_path))
            print(f"Extracted: {pdf_path.name}")
        except Exception as exc:
            print(f"Could not read {pdf_path.name}: {exc}")

    return all_pages


if __name__ == "__main__":
    pages = extract_all_pdfs()

    print(f"\nExtracted text from {len(pages)} non-empty pages.")

    if pages:
        print("\nSample extracted text:")
        print(pages[0]["text"][:1000])