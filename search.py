import re
import sqlite3

# The database layer (tables, chunking, saving) lives in database.py.
# These names are re-exported so older imports from search.py keep working.
from database import connect, initialize_database, save_page  # noqa: F401


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "could", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "me", "of", "on", "or",
    "please", "tell", "that", "the", "their", "them",
    "there", "these", "this", "to", "was", "we", "what",
    "when", "where", "which", "who", "why", "with",
    "would", "you", "your", "aesl", "sell", "selling",
    "sold", "offer", "offers", "offering", "provide",
    "provides", "available", "availability", "have",
    "has", "stock", "carry", "supplies", "supply",
}

MAX_CHUNKS_PER_PAGE = 2


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def stem(word):
    """Very light plural stripping, matching what the FTS porter index does."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def build_search_terms(question):
    words = tokenize(question)

    return list(dict.fromkeys(
        stem(word) for word in words
        if len(word) > 1 and word not in STOP_WORDS
    ))


def search_pages(question, limit=5):
    """
    Return the most relevant website chunks as dicts with
    url, title and content (one chunk per result).
    """
    terms = build_search_terms(question)

    if not terms:
        return []

    fts_query = " OR ".join(f'"{term}"' for term in terms[:15])

    try:
        with connect() as conn:
            rows = conn.execute("""
                SELECT
                    url,
                    title,
                    content,
                    bm25(chunk_search, 0.0, 5.0, 1.0) AS rank
                FROM chunk_search
                WHERE chunk_search MATCH ?
                ORDER BY rank ASC
                LIMIT 40
            """, (fts_query,)).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"Search error: {exc}")
        return []

    candidates = []

    for row in rows:
        page = dict(row)

        title_terms = {stem(w) for w in tokenize(page["title"])}
        content_terms = {stem(w) for w in tokenize(page["content"])}

        title_matches = sum(term in title_terms for term in terms)
        matched = sum(
            term in title_terms or term in content_terms
            for term in terms
        )

        # One-word questions need that word; longer ones need two distinct
        # query terms in the same chunk, so a stray word isn't enough.
        if matched < min(2, len(terms)):
            continue

        page["_relevance"] = title_matches * 3 + matched
        candidates.append(page)

    # More matched terms (title matches count extra) first, then BM25 rank
    # (lower is better).
    candidates.sort(
        key=lambda page: (page["_relevance"], -page["rank"]),
        reverse=True,
    )

    results = []
    per_page = {}

    for page in candidates:
        count = per_page.get(page["url"], 0)

        if count >= MAX_CHUNKS_PER_PAGE:
            continue

        per_page[page["url"]] = count + 1
        page.pop("_relevance", None)
        page.pop("rank", None)
        results.append(page)

        if len(results) >= limit:
            break

    return results


if __name__ == "__main__":
    initialize_database()

    print("AESL website search test")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nEnter a search question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        results = search_pages(question)

        if not results:
            print("No sufficiently relevant results found.")
            continue

        for i, result in enumerate(results, start=1):
            print(f"\n--- Result {i} ---")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Content: {result['content']}")
