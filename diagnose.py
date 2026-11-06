"""
Shows what the assistant can actually see and what it finds for a question,
without calling the language model.

    python diagnose.py "your question here"
"""
import sqlite3
import sys


def section(title):
    print(f"\n=== {title} ===")


question = " ".join(sys.argv[1:]).strip()

if not question:
    question = input("Question to test: ").strip()

# ---------------------------------------------------------------- Ollama
section("Ollama")
try:
    import ollama
    from config import OLLAMA_MODEL

    installed = []

    for model in ollama.list().models:
        name = getattr(model, "model", None) or getattr(model, "name", "")
        installed.append(name)

    print("Running. Installed models:", ", ".join(installed) or "none")

    if not any(name.startswith(OLLAMA_MODEL) for name in installed):
        print(f"PROBLEM: '{OLLAMA_MODEL}' is not installed. "
              f"Run: ollama pull {OLLAMA_MODEL}")
    else:
        print(f"OK: '{OLLAMA_MODEL}' is installed.")
except Exception as exc:
    print(f"PROBLEM: can't reach Ollama ({exc}). Start Ollama first.")

# ------------------------------------------------------------ embeddings
section("Meaning-based search (embeddings)")
try:
    from config import EMBED_MIN_SIMILARITY, EMBED_MODEL, PDF_ROUTE_SIMILARITY
    from embeddings import embed

    if embed(["test"]) is None:
        print(f"Not available: run  ollama pull {EMBED_MODEL}  "
              "(PDF search will use keywords only until then).")
    else:
        print(f"OK: {EMBED_MODEL} works. Minimum similarity "
              f"{EMBED_MIN_SIMILARITY}, PDF routing at {PDF_ROUTE_SIMILARITY}.")
except Exception as exc:
    print(f"PROBLEM: {exc}")

# ------------------------------------------------------ website database
section("Website database")
try:
    from config import DATABASE_PATH
    from database import database_stats

    if not DATABASE_PATH.exists():
        print("PROBLEM: no website database. Run: python scraper.py")
    else:
        stats = database_stats()
        print(f"Pages stored: {stats['pages']}, "
              f"searchable chunks: {stats['chunks']}, "
              f"last update: {stats['latest_update']}")

        from database import connect

        with connect() as conn:
            embedded = conn.execute(
                "SELECT COUNT(*) FROM chunk_vectors"
            ).fetchone()[0]
            contact = conn.execute(
                "SELECT 1 FROM pages WHERE url LIKE '%#contact-details'"
            ).fetchone()

        print(f"Chunks with embeddings: {embedded} of {stats['chunks']}"
              + ("" if embedded else "  (meaning-based search inactive)"))
        print("Contact-details entry: "
              + ("saved" if contact else "missing (re-run scraper.py)"))

        if not stats["pages"]:
            print("PROBLEM: the database is empty. "
                  "Run: python scraper.py and check its summary.")
except Exception as exc:
    print(f"PROBLEM: {exc}")

# ------------------------------------------------------------ PDF index
section("PDF index")
try:
    from document_reader import DOCUMENTS_DIR
    from document_search import connect_db, ensure_pdf_index
    from document_tables import connect_db as table_db

    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    print(f"PDFs in documents/: {[p.name for p in pdfs] or 'NONE'}")

    if not pdfs:
        print("PROBLEM: put your PDFs in the documents/ folder.")

    ensure_pdf_index()

    with connect_db() as conn:
        for row in conn.execute(
            "SELECT filename, COUNT(*) AS n FROM pdf_chunks GROUP BY filename"
        ):
            print(f"  passages from {row['filename']}: {row['n']}")

    with table_db() as conn:
        for row in conn.execute(
            "SELECT filename, COUNT(*) AS n FROM pdf_table_rows "
            "GROUP BY filename"
        ):
            print(f"  table rows from {row['filename']}: {row['n']}")
except Exception as exc:
    print(f"PROBLEM: {exc}")

# ------------------------------------------------------ what the question hits
section(f"Question: {question}")

try:
    from document_tables import is_direct_table_lookup, search_table_rows

    rows = search_table_rows(question, limit=5)
    print(f"\nTable rows found: {len(rows)}"
          f"  (direct lookup: {is_direct_table_lookup(question, rows)})")

    for row in rows:
        print(f"  {row['filename']} p{row['page']} score={row['score']:.1f} "
              f"matched={sorted(set(row['matched_terms']))}")
        print("     ", " | ".join(row["values"])[:150])
except Exception as exc:
    print(f"PROBLEM in table search: {exc}")

try:
    from document_search import search_documents

    passages = search_documents(question, limit=4)
    print(f"\nPDF passages found: {len(passages)}")

    for passage in passages:
        sim = passage.get("similarity")
        label = f" similarity={sim:.2f}" if sim else ""
        print(f"  {passage['filename']} p{passage['page']}{label}")
        print("     ", passage["text"][:200].replace("\n", " / "))
except Exception as exc:
    print(f"PROBLEM in PDF search: {exc}")

try:
    from search import search_pages

    pages = search_pages(question, limit=4)
    print(f"\nWebsite chunks found: {len(pages)}")

    for page in pages:
        sim = page.get("similarity")
        label = f" similarity={sim:.2f}" if sim else ""
        print(f"  {page['title']} - {page['url']}{label}")
        print("     ", page["content"][:200].replace("\n", " / "))
except Exception as exc:
    print(f"PROBLEM in website search: {exc}")

print()