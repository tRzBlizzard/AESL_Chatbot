import re
import sqlite3
from pathlib import Path

import numpy as np

from config import EMBED_MIN_SIMILARITY, EMBED_MODEL, SEARCH_SYNONYMS
from document_reader import (
    extract_all_pdfs,
    looks_like_heading,
    looks_like_soft_heading,
    pdf_signature,
)
from document_tables import rebuild_table_index, table_text_blocks
from embeddings import embed

BASE_DIR = Path(__file__).resolve().parent
PDF_DB = BASE_DIR / "pdf_documents.db"

# Chunking settings: split long pages into overlapping passages.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

def stem(word):
    """Light plural stripping, close to what the FTS porter index does."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"

    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


STOP_WORDS = {
    "only", "just", "also", "details", "detail", "information", "info",
    "regarding", "give", "show", "listed", "mentioned", "provided",
    "document", "documents", "pdf", "kindly", "want", "need", "know",
    "much", "many", "did", "will", "was", "were", "has", "have", "had",
    "difference", "differences", "different", "between", "types", "type",
    "kinds", "kind", "compare", "comparison", "versus", "vs", "various",
    "heard", "people", "mention", "here", "them", "they", "their",
    "ve", "ll", "re",
    "what", "is", "are", "the", "a", "an", "of", "in",
    "on", "for", "to", "and", "does", "do", "how",
    "can", "explain", "describe", "tell", "me", "about",
    "i", "my", "we", "our", "you", "your", "there", "which",
    "who", "when", "why", "would", "should", "any", "with",
    "please", "get", "if", "be", "it", "this", "that",
}


NUMBERED_ITEM = re.compile(r"^\s*(\d{1,2})[.)]\s+\S")


def _item_number(line):
    match = NUMBERED_ITEM.match(line)
    return int(match.group(1)) if match else None


def is_section_heading(lines, index):
    """
    A heading line: not a column note like 'NOT EXCEEDING' above a rate row,
    not the last word or two of a wrapped sentence, and not an item of a
    numbered list ("1. Zeeshan Kazmi", "2. Ifran Burney").
    """
    line = lines[index]
    previous = lines[index - 1] if index > 0 else None
    following = lines[index + 1] if index + 1 < len(lines) else ""

    number = _item_number(line)

    if number is not None:
        before = _item_number(previous) if previous else None
        after = _item_number(following)

        if before == number - 1 or after == number + 1:
            return False

    if looks_like_soft_heading(line, previous, following):
        return True

    if not looks_like_heading(line, previous):
        return False

    return not re.match(
        r"^\d{1,2}\s*(?:to\b|-|\u2013|\d|rs\b)|^rs\b", following, re.I
    )


def wrap_long_line(line):
    """Break an over-long line into pieces at spaces."""
    pieces = []

    while len(line) > CHUNK_SIZE:
        cut = line.rfind(" ", 0, CHUNK_SIZE)
        cut = cut if cut > CHUNK_SIZE // 2 else CHUNK_SIZE
        pieces.append(line[:cut].strip())
        line = line[cut:].strip()

    if line:
        pieces.append(line)

    return pieces


def clean_page_lines(text):
    """Tidy one page's text into lines (bullets removed, breaks repaired)."""
    lines = []

    for raw_line in text.splitlines():
        # Bullets from symbol fonts (e.g. "\uf06c") are layout, not content.
        line = re.sub(r"[\ue000-\uf8ff]", " ", raw_line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = re.sub(r"^[\u2022\u25aa\u25cf\u25e6\u2023\u00b7]\s*", "", line)

        if line:
            lines.extend(wrap_long_line(line))

    # "...(for the period to 31 Dec" / "2026)": a bracket cut by a line break.
    merged = []

    for line in lines:
        if (
            merged
            and merged[-1].count("(") > merged[-1].count(")")
            and re.fullmatch(r"[^()]{1,20}\)", line)
        ):
            merged[-1] += " " + line
        else:
            merged.append(line)

    return merged


def chunk_document(pages):
    """
    Split ONE document (its pages, in order) into chunks. A new chunk starts
    at each section heading, so a section's rules, rows and list stay
    together, and sections run straight across page breaks: a list that
    continues on the next page, or a heading at the bottom of a page with
    its items on the next, still belong together. Long sections are split
    with some overlap, and every chunk is labelled with its section.

    Returns (chunks, lines): chunks is [(page label, text)], and lines is
    [(page, line, section in effect)] for locating things in the document.
    """
    lines = []

    for item in pages:
        for line in clean_page_lines(item["text"]):
            lines.append((line, item["page"]))

    texts = [line for line, _ in lines]
    chunks = []
    line_info = []
    current = []   # [line, page, section, is_heading, is_overlap]
    size = 0
    heading = ""

    def emit():
        # A heading stranded on its own.
        if len(current) == 1 and current[0][3]:
            return

        body = "\n".join(item[0] for item in current)
        label = next(
            (item[0] for item in reversed(current) if item[3]), ""
        ) or current[0][2]

        if label and not body.startswith(label):
            body = f"[Section: {label}]\n{body}"

        pages_in = sorted({item[1] for item in current if not item[4]})

        if not pages_in:
            return

        page = (
            pages_in[0] if len(pages_in) == 1
            else f"{pages_in[0]}-{pages_in[-1]}"
        )
        chunks.append((page, body))

    for index, (line, page) in enumerate(lines):
        is_head = is_section_heading(texts, index)

        if is_head:
            heading = line

            if any(not item[3] and not item[4] for item in current):
                emit()

            current = []
            size = 0
        elif current and size + len(line) + 1 > CHUNK_SIZE:
            emit()

            # Start the next chunk with the tail of this one (overlap).
            tail = []
            tail_size = 0

            for item in reversed(current):
                if tail_size + len(item[0]) + 1 > CHUNK_OVERLAP:
                    break

                tail.insert(0, item[:4] + [True])
                tail_size += len(item[0]) + 1

            current = tail
            size = tail_size

        current.append([line, page, heading, is_head, False])
        size += len(line) + 1
        line_info.append((page, line, heading))

    if current:
        emit()

    return chunks, line_info


def split_into_chunks(text):
    """Chunks of a single block of text (see chunk_document)."""
    chunks, _ = chunk_document([{"page": 1, "text": text}])
    return [body for _, body in chunks]


def section_of(line_info, page, snippet):
    """The section a piece of text sits under (e.g. a table row's names)."""
    wanted = re.sub(r"^\d+[.)]\s*|\s*\d+[.)]$", "", snippet).strip().lower()

    if not wanted:
        return ""

    for line_page, line, heading in line_info:
        if line_page == page and wanted in line.lower():
            return heading

    return ""


def connect_db():
    conn = sqlite3.connect(PDF_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


CREATE_CHUNKS_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS pdf_chunks
    USING fts5(
        filename,
        page UNINDEXED,
        text,
        tokenize='porter unicode61'
    )
"""


CREATE_VECTORS_SQL = """
    CREATE TABLE IF NOT EXISTS pdf_vectors (
        chunk_id INTEGER PRIMARY KEY,
        vector BLOB NOT NULL
    )
"""

# Loaded once per index version: (chunk ids, matrix of unit vectors).
_vector_cache = None


def initialize_index():
    """Create the searchable PDF index if it doesn't exist."""
    with connect_db() as conn:
        conn.execute(CREATE_CHUNKS_SQL)
        conn.execute(CREATE_VECTORS_SQL)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)


def build_vectors():
    """
    Embed every passage so questions can be matched by meaning. Safe to
    call again; does nothing useful if the embedding model isn't available.
    """
    global _vector_cache
    _vector_cache = None

    with connect_db() as conn:
        conn.execute(CREATE_VECTORS_SQL)
        conn.execute("DELETE FROM pdf_vectors")
        conn.execute("DELETE FROM index_meta WHERE key = 'embed_model'")

        rows = conn.execute(
            "SELECT rowid AS id, filename, text FROM pdf_chunks ORDER BY rowid"
        ).fetchall()

        if not rows:
            return False

        print(f"Embedding {len(rows)} passages with {EMBED_MODEL}...")
        matrix = embed([f"{r['filename']}: {r['text']}" for r in rows])

        if matrix is None:
            return False

        conn.executemany(
            "INSERT INTO pdf_vectors (chunk_id, vector) VALUES (?, ?)",
            [(r["id"], matrix[i].tobytes()) for i, r in enumerate(rows)],
        )
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) "
            "VALUES ('embed_model', ?)",
            (EMBED_MODEL,),
        )

    print("Meaning-based search is ready.")
    return True


def load_vectors():
    global _vector_cache

    if _vector_cache is None:
        with connect_db() as conn:
            rows = conn.execute(
                "SELECT chunk_id, vector FROM pdf_vectors ORDER BY chunk_id"
            ).fetchall()

        if rows:
            ids = [r["chunk_id"] for r in rows]
            matrix = np.vstack([
                np.frombuffer(r["vector"], dtype=np.float32) for r in rows
            ])
            _vector_cache = (ids, matrix)
        else:
            _vector_cache = ([], None)

    return _vector_cache


def vector_search(question, limit):
    """[(chunk_id, similarity)] for passages close in meaning to the question."""
    ids, matrix = load_vectors()

    if matrix is None:
        return []

    query = embed([question], kind="query")

    if query is None or query.shape[1] != matrix.shape[1]:
        return []

    similarity = matrix @ query[0]
    best = np.argsort(-similarity)[:limit]

    return [
        (ids[i], float(similarity[i]))
        for i in best
        if similarity[i] >= EMBED_MIN_SIMILARITY
    ]


def rebuild_index():
    """Re-extract PDFs and rebuild both PDF indexes from scratch."""
    # Structured table rows first: their text is added to the passages too.
    rebuild_table_index()

    pages = extract_all_pdfs()
    table_blocks = table_text_blocks()
    signature = pdf_signature()
    _modified_cache.clear()

    with connect_db() as conn:
        # Dropped and recreated so older index layouts are replaced.
        conn.execute("DROP TABLE IF EXISTS pdf_chunks")
        conn.execute("DROP TABLE IF EXISTS pdf_vectors")
        conn.execute(CREATE_CHUNKS_SQL)
        conn.execute(CREATE_VECTORS_SQL)
        conn.execute("DELETE FROM index_meta WHERE key = 'embed_model'")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        rows = []
        documents = {}

        for item in pages:
            documents.setdefault(item["document"], []).append(item)

        line_infos = {}

        for document, doc_pages in documents.items():
            chunks, line_infos[document] = chunk_document(doc_pages)

            for page, chunk in chunks:
                rows.append((document, page, chunk))

        conn.executemany(
            "INSERT INTO pdf_chunks (filename, page, text) VALUES (?, ?, ?)",
            rows,
        )

        # Table rows are indexed as extra passages after the text ones. A
        # table whose heading sits on the previous page (a list of names
        # under "Allied Blood Donation Team") gets that heading too.
        text_end = conn.execute(
            "SELECT COALESCE(MAX(rowid), 0) FROM pdf_chunks"
        ).fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) "
            "VALUES ('text_chunks_end', ?)",
            (str(text_end),),
        )

        block_rows = []

        for block in table_blocks:
            text = block["text"]
            section = section_of(
                line_infos.get(block["document"], []),
                block["page"],
                block.get("first_value", ""),
            )

            if section and section not in text:
                text = f"[Section: {section}]\n{text}"

            block_rows.append((block["document"], block["page"], text))

        conn.executemany(
            "INSERT INTO pdf_chunks (filename, page, text) VALUES (?, ?, ?)",
            block_rows,
        )
        rows += block_rows

        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) "
            "VALUES ('pdf_signature', ?)",
            (signature,),
        )

    print(f"Indexed {len(pages)} PDF pages ({len(rows)} passages).")
    print(f"Database location: {PDF_DB}")

    build_vectors()


def indexed_documents():
    try:
        with connect_db() as conn:
            return [
                row[0] for row in conn.execute(
                    "SELECT DISTINCT filename FROM pdf_chunks"
                )
            ]
    except sqlite3.OperationalError:
        return []


def _same_word(a, b):
    """'function' ~ 'functional', 'team' ~ 'teams'."""
    shared = 0

    for x, y in zip(a, b):
        if x != y:
            break

        shared += 1

    return shared >= max(4, int(0.8 * min(len(a), len(b))))


def match_document(question):
    """
    The indexed PDF the question names (e.g. "cross function teams
    document" -> "Cross Functional Teams.pdf"), or None.
    """
    asked = [w for w in re.findall(r"[a-z]+", question.lower()) if len(w) >= 4]
    best, best_hits = None, 0

    for filename in indexed_documents():
        name_words = [
            w for w in re.findall(r"[a-z]+", filename.lower().rsplit(".", 1)[0])
            if len(w) >= 4
        ]

        if not name_words:
            continue

        hits = sum(
            any(_same_word(name, word) for word in asked)
            for name in name_words
        )

        enough = hits >= 2 or (len(name_words) == 1 and hits == 1)

        if enough and hits > best_hits:
            best, best_hits = filename, hits

    return best


def document_chunks(filename, max_chars=10000):
    """
    A document's own passages in reading order (table passages excluded, as
    they repeat the text), as many as fit in max_chars. Returns
    (passages, whether anything was left out).
    """
    with connect_db() as conn:
        end = conn.execute(
            "SELECT value FROM index_meta WHERE key = 'text_chunks_end'"
        ).fetchone()
        rows = conn.execute(
            "SELECT rowid AS id, filename, page, text FROM pdf_chunks "
            "WHERE filename = ? AND rowid <= ? ORDER BY rowid",
            (filename, int(end["value"]) if end else 10**9),
        ).fetchall()

    passages, used = [], 0

    for row in rows:
        if used + len(row["text"]) > max_chars and passages:
            return passages, True

        item = dict(row)
        item["similarity"] = None
        passages.append(item)
        used += len(row["text"])

    return passages, False


def document_mentions_grades(filename):
    """True if the document has employee grades (a GRADE column, 'grades 1-5')."""
    try:
        with connect_db() as conn:
            return conn.execute(
                "SELECT 1 FROM pdf_chunks WHERE filename = ? "
                "AND text LIKE '%grade%' LIMIT 1",
                (filename,),
            ).fetchone() is not None
    except sqlite3.OperationalError:
        return False


_modified_cache = {}

_MODIFIED_PATTERN = re.compile(
    r"(?i)\bLast\s+Modified\s*:\s*"
    r"(\d{1,2}\s*[-/ ]\s*[A-Za-z]{3,9}\s*[-/ ]\s*\d{2,4}"
    r"|\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
)


def get_pdf_last_modified(filename):
    """Return the revision date printed in a PDF, if it is indexed.

    This intentionally reads the extracted PDF text, not the filesystem's
    modified timestamp. Only dates explicitly following "Last Modified" are
    accepted. Returns the date text as written, or None when absent.
    """
    if filename in _modified_cache:
        return _modified_cache[filename]

    found = None

    try:
        with connect_db() as conn:
            rows = conn.execute(
                "SELECT text FROM pdf_chunks "
                "WHERE filename = ? AND text LIKE '%Last Modified%' "
                "ORDER BY rowid LIMIT 50",
                (filename,),
            ).fetchall()
    except sqlite3.OperationalError:
        # The PDF index may not have been initialized yet.
        return None

    for row in rows:
        match = _MODIFIED_PATTERN.search(row["text"] or "")

        if match:
            found = re.sub(r"\s+", "", match.group(1))
            break

    _modified_cache[filename] = found
    return found


def ensure_pdf_index():
    """
    Rebuild the PDF indexes only when the files in documents/ changed
    (or the extraction code was updated). Call this at startup.
    """
    initialize_index()

    with connect_db() as conn:
        row = conn.execute(
            "SELECT value FROM index_meta WHERE key = 'pdf_signature'"
        ).fetchone()

    if row is None or row["value"] != pdf_signature():
        print("PDF files changed - rebuilding the PDF index...")
        rebuild_index()
        return

    # PDFs unchanged, but the embeddings may be missing (e.g. the embedding
    # model was installed after the last rebuild).
    with connect_db() as conn:
        embedded = conn.execute(
            "SELECT value FROM index_meta WHERE key = 'embed_model'"
        ).fetchone()
        has_chunks = conn.execute(
            "SELECT 1 FROM pdf_chunks LIMIT 1"
        ).fetchone()

    if has_chunks and (embedded is None or embedded["value"] != EMBED_MODEL):
        build_vectors()


def keyword_search(question, limit):
    """
    Keyword matches, preferring an exact phrase, then all words, then any
    word. Words map to their SEARCH_SYNONYMS (e.g. weekend -> saturday,
    sunday). Rows from the first two kinds carry strong=True.
    """
    words = [
        w for w in re.findall(r"\w+", question.lower(), flags=re.UNICODE)
        if len(w) > 1 or w.isdigit()
    ]

    # Remove common question words that don't help identify the topic.
    keywords = list(dict.fromkeys(
        word for word in words if word not in STOP_WORDS
    ))

    if not keywords:
        keywords = list(dict.fromkeys(words))

    if not keywords:
        return []

    keywords = keywords[:12]

    def quote(word):
        return '"' + word.replace('"', '""') + '"'

    # keyword -> alternative words; synonym (stemmed) -> the keyword it means
    alternatives = {}
    means = {}

    for word in keywords:
        extra = [
            alt for alt in SEARCH_SYNONYMS.get(word, [])
            if alt not in keywords
        ]
        alternatives[word] = extra

        for alt in extra:
            means[stem(alt)] = stem(word)

    def group(word):
        options = [word] + alternatives[word]

        if len(options) == 1:
            return quote(word)

        return "(" + " OR ".join(quote(option) for option in options) + ")"

    sql = """
        SELECT rowid AS id, filename, page, text, bm25(pdf_chunks) AS score
        FROM pdf_chunks
        WHERE pdf_chunks MATCH ?
        ORDER BY score ASC
        LIMIT ?
    """

    queries = []

    # 1. Prefer an exact phrase when the question has multiple keywords.
    if len(keywords) >= 2:
        queries.append(quote(" ".join(keywords)))

    # 2. Require every meaningful keyword (or a synonym of it).
    queries.append(" AND ".join(group(word) for word in keywords))

    # 3. Fall back to broader matching only if stricter searches fail.
    queries.append(" OR ".join(
        quote(option)
        for word in keywords
        for option in [word] + alternatives[word]
    ))

    results = []
    seen = set()

    def add(rows):
        for row in rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                results.append(row)

    with connect_db() as conn:
        for position, fts_query in enumerate(queries):
            if len(results) >= limit:
                break

            is_any_word_query = position == len(queries) - 1

            try:
                rows = conn.execute(
                    sql,
                    (fts_query, limit * 4 if is_any_word_query else limit),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                print(f"PDF search warning: {exc}")
                continue

            rows = [dict(row) for row in rows]

            if not is_any_word_query:
                for row in rows:
                    row["strong"] = True

                add(rows)
                continue

            # "Any word" matches are noisy: require at least two of the
            # question's distinct words when it has several. Short questions
            # often use words the document doesn't ("family" vs "spouse"),
            # so only longer ones need two hits.
            wanted = {stem(word) for word in keywords}
            needed = 1 if len(wanted) <= 3 else 2

            for row in rows:
                present = {
                    means.get(stem(word), stem(word))
                    for word in re.findall(r"\w+", row["text"].lower())
                }
                row["matched"] = len(wanted & present)
                row["strong"] = (
                    row["matched"] >= max(2, -(-len(wanted) * 3 // 5))
                )

            # BM25 (score) already ranks rare, telling words ("food") above
            # common ones ("allowance"), so keep that order.
            add([row for row in rows if row["matched"] >= needed])

    return results[:limit]


def search_documents(question, limit=3):
    """
    Retrieve focused PDF passages by combining keyword matches (good for
    exact words, names, numbers) with meaning-based matches (good for
    different wording). Each result may carry a "similarity" value when it
    was found by meaning.
    """
    initialize_index()

    candidates = max(limit * 2, 8)
    keyword_rows = keyword_search(question, candidates)
    vector_hits = vector_search(question, candidates)

    if not vector_hits:
        for row in keyword_rows:
            row["similarity"] = None

        return keyword_rows[:limit]

    # Reciprocal rank fusion: a passage ranked well by either method (and
    # especially by both) comes out on top.
    fused = {}
    similarity = {}
    rows_by_id = {row["id"]: row for row in keyword_rows}

    for rank, row in enumerate(keyword_rows):
        fused[row["id"]] = fused.get(row["id"], 0) + 1 / (60 + rank)

    for rank, (chunk_id, sim) in enumerate(vector_hits):
        fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (60 + rank)
        similarity[chunk_id] = sim

    missing = [cid for cid in fused if cid not in rows_by_id]

    if missing:
        marks = ",".join("?" for _ in missing)

        with connect_db() as conn:
            for row in conn.execute(
                f"SELECT rowid AS id, filename, page, text FROM pdf_chunks "
                f"WHERE rowid IN ({marks})",
                missing,
            ):
                rows_by_id[row["id"]] = dict(row)

    results = []

    for chunk_id in sorted(fused, key=fused.get, reverse=True)[:limit]:
        row = rows_by_id.get(chunk_id)

        if row:
            row["similarity"] = similarity.get(chunk_id)
            results.append(row)

    return results


if __name__ == "__main__":
    rebuild_index()

    print("\nPDF index ready. Test it with your own questions.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        results = search_documents(question)

        if not results:
            print("No matching passages found.\n")
            continue

        for result in results:
            print(
                f"\nSource: {result['filename']} "
                f"| Page: {result['page']}"
            )
            print(result["text"])

        print()