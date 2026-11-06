import json
import re
import sqlite3
from pathlib import Path

import pymupdf

from document_reader import looks_like_heading

try:
    from rapidfuzz import fuzz, process
except ImportError:  # fuzzy name matching is optional
    fuzz = process = None

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
PDF_DB = BASE_DIR / "pdf_documents.db"

# Increase this if words that belong to separate columns are being merged.
COLUMN_GAP = 18.0

# Adjust this if words on the same visual line are being split into rows.
ROW_Y_TOLERANCE = 3.5

# Minimum similarity (0-100) for a misspelt name to still count as a match,
# e.g. "Hassan" vs "Hasan".
FUZZY_CUTOFF = 86

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "how", "i", "in", "is", "it", "me",
    "of", "on", "or", "please", "tell", "that", "the", "their", "them",
    "there", "these", "this", "to", "was", "we", "what", "when", "where",
    "which", "who", "why", "with", "would", "you", "your", "about", "find",
    "show", "give", "get", "list", "lookup", "look", "search", "employee",
    "employees", "person", "people", "details", "detail", "information",
    "has", "have", "had", "did", "will", "offer", "offers", "offered",
    "offering", "provide", "provides", "provided", "sell", "sold",
    "named", "called", "extension", "ext", "phone", "telephone", "number",
    "numbers", "region", "designation", "price", "cost", "amount", "stock",
    "available", "availability", "quantity", "part", "parts", "product",
    "products", "email", "address", "contact", "contacts", "department",
    "office", "serial", "model",
}

HEADER_ALIASES = {
    "name": {
        "name", "employee", "employee name", "staff name", "person",
        "contact name", "employee names",
    },
    "designation": {
        "designation", "title", "job title", "position", "post",
    },
    "extension": {
        "extension", "ext", "extn", "ext no", "extension no",
        "extension number", "ext number", "extn no",
    },
    "phone": {
        "phone", "telephone", "mobile", "cell", "contact no",
        "phone no", "telephone no", "mobile no", "number",
        "phone number", "telephone number",
    },
    "region": {
        "region", "city", "location", "branch", "area", "office",
    },
    "department": {"department", "dept"},
    "email": {"email", "e mail", "email address"},
}


BULLET_GLYPHS = re.compile(r"^[\u2022\u25aa\u25cf\u25e6\u2023\u00b7\s]+")


def clean_cell(value):
    """Normalize whitespace and drop bullet glyphs from a cell."""
    text = str(value or "")
    text = re.sub(r"[\ue000-\uf8ff]", " ", text)  # symbol-font bullets
    text = re.sub(r"\s+", " ", text).strip()
    return BULLET_GLYPHS.sub("", text).strip()


def normalize_header(value):
    value = clean_cell(value).lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def canonical_header(value):
    cleaned = normalize_header(value)

    for canonical, aliases in HEADER_ALIASES.items():
        normalized_aliases = {normalize_header(a) for a in aliases}
        if cleaned in normalized_aliases:
            return canonical.title()

    return clean_cell(value)


def tokens(text):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    result = []

    for word in words:
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]

        if len(word) > 1 or word.isdigit():
            result.append(word)

    return result


def query_tokens(question):
    return list(dict.fromkeys(
        token for token in tokens(question)
        if token not in STOP_WORDS
    ))


def connect_db():
    conn = sqlite3.connect(PDF_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def initialize_table_index():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pdf_table_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                page INTEGER NOT NULL,
                table_number INTEGER NOT NULL,
                row_number INTEGER NOT NULL,
                headers_json TEXT NOT NULL,
                values_json TEXT NOT NULL,
                context TEXT NOT NULL,
                searchable_text TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pdf_table_source
            ON pdf_table_rows(filename, page, table_number)
        """)


def extract_layout_lines(page):
    """
    Reconstruct visual lines and cells from PDF word coordinates.

    This supports text-based PDFs whose columns are separated by whitespace,
    rather than actual Excel cells or bordered tables.
    """
    try:
        words = page.get_text("words", sort=True)
    except Exception:
        return []

    positioned_words = []

    for word in words:
        # PyMuPDF word tuples contain x0, y0, x1, y1, text, ...
        if len(word) < 5:
            continue

        try:
            x0, y0, x1, y1 = map(float, word[:4])
        except (TypeError, ValueError):
            continue

        text = clean_cell(word[4])
        if text:
            positioned_words.append({
                "x0": x0,
                "x1": x1,
                "y": (y0 + y1) / 2,
                "text": text,
            })

    positioned_words.sort(key=lambda item: (item["y"], item["x0"]))

    # Group words with approximately the same vertical position.
    lines = []

    for word in positioned_words:
        matching_line = None

        # Search recent lines, since neighboring words can have slightly
        # different vertical centers in some PDF fonts.
        for line in reversed(lines[-3:]):
            if abs(word["y"] - line["y"]) <= ROW_Y_TOLERANCE:
                matching_line = line
                break

        if matching_line is None:
            lines.append({"y": word["y"], "words": [word]})
        else:
            matching_line["words"].append(word)
            matching_line["y"] = (
                matching_line["y"] * (len(matching_line["words"]) - 1)
                + word["y"]
            ) / len(matching_line["words"])

    output = []

    for line in lines:
        line_words = sorted(line["words"], key=lambda item: item["x0"])
        cells = []
        current_words = []
        previous_x1 = None

        for word in line_words:
            gap = (
                word["x0"] - previous_x1
                if previous_x1 is not None
                else 0
            )

            if current_words and gap >= COLUMN_GAP:
                cells.append({
                    "text": " ".join(w["text"] for w in current_words),
                    "x0": current_words[0]["x0"],
                })
                current_words = []

            current_words.append(word)
            previous_x1 = word["x1"]

        if current_words:
            cells.append({
                "text": " ".join(w["text"] for w in current_words),
                "x0": current_words[0]["x0"],
            })

        cells = [
            cell for cell in cells
            if clean_cell(cell["text"])
        ]

        if cells:
            output.append(cells)

    return output


def looks_like_header(cells):
    recognized = 0

    for cell in cells:
        header = normalize_header(canonical_header(cell["text"]))

        if header in HEADER_ALIASES:
            recognized += 1

    return recognized >= 2


def has_known_headers(headers):
    """True for headers like Name / Designation / Extension."""
    return sum(
        normalize_header(header) in HEADER_ALIASES for header in headers
    ) >= 2


def table_context(recent_lines):
    """The heading and last sentence above a table, e.g. 'Food Allowance'."""
    heading = next(
        (line for line in reversed(recent_lines) if looks_like_heading(line)),
        "",
    )
    last = recent_lines[-1] if recent_lines else ""

    parts = [heading, last[:160] if last != heading else ""]
    return " - ".join(part for part in parts if part)


def extract_layout_table_rows(page, inherited_headers=None):
    """
    Infer table rows from whitespace-separated columns.

    A page can hold several small tables between paragraphs (e.g. a hotel
    rates table and, further down, a food rates table), so a table ends
    when ordinary paragraph text starts, and the next table gets its own
    headers and its own heading as context. Directory-style headers
    (Name / Designation / Extension ...) carry over to continuation pages.
    """
    lines = extract_layout_lines(page)

    if not lines:
        return [], inherited_headers or []

    reusable = (
        list(inherited_headers)
        if inherited_headers and has_known_headers(inherited_headers)
        else []
    )
    headers = []
    context = ""
    output = []
    table_number = 0
    row_number = 0
    in_table = False
    prose_run = 0
    recent = []  # headings and paragraph lines seen since the last table

    for cells in lines:
        values = [clean_cell(cell["text"]) for cell in cells]
        values = [value for value in values if value]

        if not values:
            continue

        text = " ".join(values)
        word_count = len(text.split())

        # A full sentence on its own line: paragraph text, not a table row.
        # (A single wrapped cell is short; a paragraph is two lines or one
        # long one.)
        if len(values) == 1 and word_count >= 5:
            prose_run += 1
            recent = (recent + [text])[-8:]

            if in_table and (prose_run >= 2 or word_count >= 10):
                in_table = False

            continue

        prose_run = 0

        # Headings, page numbers and short notes ("NOT EXCEEDING") aren't rows.
        if len(values) < 2:
            if not in_table:
                recent = (recent + [text])[-8:]

            continue

        if not in_table:
            in_table = True
            table_number += 1
            row_number = 0
            context = table_context(recent)
            recent = []
            headers = []

            # A directory continuing without a repeated header row.
            if (
                reusable
                and len(values) == len(reusable)
                and not looks_like_header(cells)
            ):
                headers = list(reusable)

        # A recognised header row (also repeated headers on later pages).
        if looks_like_header(cells):
            headers = [
                canonical_header(cell["text"]) or f"Column {n + 1}"
                for n, cell in enumerate(cells)
            ]
            reusable = list(headers)
            continue

        # No recognised header (e.g. "GRADE | HOTEL STAY | MEALS"): the
        # first all-text row is the header, not a record.
        if not headers and not any(re.search(r"\d", v) for v in values):
            headers = list(values)
            continue

        if not headers:
            headers = [f"Column {n + 1}" for n in range(len(values))]

        # If the number of detected cells differs from the header count,
        # retain every detected cell and add generic columns as necessary.
        column_count = max(len(headers), len(values))

        if len(headers) < column_count:
            headers.extend(
                f"Column {n + 1}"
                for n in range(len(headers), column_count)
            )

        values.extend([""] * (column_count - len(values)))
        values = values[:column_count]

        row_number += 1

        output.append({
            "table_number": table_number,
            "row_number": row_number,
            "headers": list(headers[:column_count]),
            "values": values,
            "context": context,
        })

    return output, (reusable or headers)


def nearby_context(page, bbox):
    try:
        table_top = float(bbox[1])
    except (TypeError, ValueError, IndexError):
        return ""

    candidates = []

    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue

        try:
            block_bottom = float(block[3])
        except (TypeError, ValueError):
            continue

        text = clean_cell(block[4])

        if (
            text
            and block_bottom <= table_top
            and table_top - block_bottom <= 180
        ):
            candidates.append((block_bottom, text))

    candidates.sort(key=lambda item: item[0])
    return " ".join(text for _, text in candidates[-3:])


def extract_page_tables(page, inherited_headers=None):
    """
    First try PyMuPDF's regular table detector.
    If it detects no usable tables, reconstruct whitespace-separated rows.
    """
    output = []

    if hasattr(page, "find_tables"):
        try:
            finder = page.find_tables()
        except Exception:
            finder = None

        tables = getattr(finder, "tables", []) if finder else []

        for table_number, table in enumerate(tables, start=1):
            try:
                raw = table.extract() or []
            except Exception:
                continue

            raw = [
                [clean_cell(cell) for cell in row]
                for row in raw
                if row and any(clean_cell(cell) for cell in row)
            ]

            if not raw:
                continue

            header_obj = getattr(table, "header", None)

            try:
                header_names = (
                    [clean_cell(v) for v in (header_obj.names or [])]
                    if header_obj else []
                )
            except Exception:
                header_names = []

            external_header = (
                bool(getattr(header_obj, "external", False))
                if header_obj else False
            )

            if header_names and any(header_names):
                headers = [canonical_header(v) for v in header_names]
                data_rows = raw if external_header else raw[1:]

                if (
                    data_rows
                    and [v.lower() for v in data_rows[0]]
                    == [v.lower() for v in headers]
                ):
                    data_rows = data_rows[1:]
            else:
                headers = [canonical_header(v) for v in raw[0]]
                data_rows = raw[1:]

            # A table continued from the previous page often has no header
            # row, so its first data row would be mistaken for the header.
            # A purely numeric cell in that row shows it is data.
            if (
                inherited_headers
                and len(inherited_headers) == len(raw[0])
                and any(
                    re.fullmatch(r"[\d\s,.\-/+]+", value)
                    for value in raw[0] if value
                )
            ):
                headers = list(inherited_headers)
                data_rows = raw

            if not headers or not any(headers):
                continue

            column_count = max(
                [len(headers)] + [len(row) for row in data_rows]
            )

            headers.extend(
                f"Column {n + 1}"
                for n in range(len(headers), column_count)
            )
            headers = headers[:column_count]

            try:
                context = nearby_context(page, table.bbox)
            except Exception:
                context = ""

            # Merged cells (e.g. one Region spanning several employees)
            # come out blank on the following rows: carry the value down.
            fill_columns = [
                index for index, header in enumerate(headers)
                if header in ("Region", "Department")
            ]
            last_seen = {}

            for row_number, row in enumerate(data_rows, start=1):
                values = (row + [""] * column_count)[:column_count]

                for index in fill_columns:
                    if values[index]:
                        last_seen[index] = values[index]
                    elif any(values) and index in last_seen:
                        values[index] = last_seen[index]

                if (
                    not any(values)
                    or [v.lower() for v in values]
                    == [v.lower() for v in headers]
                ):
                    continue

                output.append({
                    "table_number": table_number,
                    "row_number": row_number,
                    "headers": headers,
                    "values": values,
                    "context": context,
                })

    if output:
        return output, output[-1]["headers"]

    # No normal tables were detected: try text-position reconstruction.
    return extract_layout_table_rows(
        page,
        inherited_headers=inherited_headers,
    )


def rebuild_table_index():
    initialize_table_index()
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in: {DOCUMENTS_DIR}")

    for pdf_path in pdf_paths:
        try:
            with pymupdf.open(pdf_path) as pdf:
                inherited_headers = []

                for page_number, page in enumerate(pdf, start=1):
                    page_items, detected_headers = extract_page_tables(
                        page,
                        inherited_headers,
                    )

                    if detected_headers:
                        inherited_headers = detected_headers

                    for item in page_items:
                        headers = item["headers"]
                        values = item["values"]
                        context = item.get("context", "")

                        searchable = " ".join([
                            pdf_path.name,
                            context,
                            " ".join(headers),
                            " ".join(values),
                        ])

                        records.append((
                            pdf_path.name,
                            page_number,
                            item["table_number"],
                            item["row_number"],
                            json.dumps(headers, ensure_ascii=False),
                            json.dumps(values, ensure_ascii=False),
                            context,
                            searchable,
                        ))

            print(f"Processed: {pdf_path.name}")

        except Exception as exc:
            print(f"Could not extract {pdf_path.name}: {exc}")

    with connect_db() as conn:
        conn.execute("DELETE FROM pdf_table_rows")

        conn.executemany("""
            INSERT INTO pdf_table_rows (
                filename, page, table_number, row_number,
                headers_json, values_json, context, searchable_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, records)

    global _ROW_CACHE
    _ROW_CACHE = None

    print(
        f"Indexed {len(records)} structured rows "
        f"from {len(pdf_paths)} PDFs."
    )
    print(f"Database location: {PDF_DB}")

    if pdf_paths and not records:
        print(
            "No rows detected. Check whether the PDFs contain selectable "
            "text. If they do, the layout may need a different column-gap "
            "or row-alignment setting."
        )


_ROW_CACHE = None
_VOCAB = set()


def load_rows():
    """Load all table rows once (parsed and tokenized), until re-indexed."""
    global _ROW_CACHE, _VOCAB

    if _ROW_CACHE is None:
        initialize_table_index()

        with connect_db() as conn:
            rows = conn.execute("""
                SELECT id, filename, page, table_number, row_number,
                       headers_json, values_json, context
                FROM pdf_table_rows
                ORDER BY id
            """).fetchall()

        cache = []

        for row in rows:
            item = dict(row)
            headers = json.loads(item.pop("headers_json"))
            values = json.loads(item.pop("values_json"))

            item["headers"] = headers
            item["values"] = values

            # Search actual cell values and nearby context. This avoids
            # matching every row just because all rows share the same headers.
            item["_tokens"] = set(tokens(" ".join([
                " ".join(values),
                item["context"] or "",
            ])))

            # Word-like tokens from the cells, for misspelt-name matching.
            item["_words"] = sorted({
                token for token in tokens(" ".join(values))
                if token.isalpha() and len(token) >= 4
            })

            cache.append(item)

        _ROW_CACHE = cache
        _VOCAB = set().union(*(item["_tokens"] for item in cache)) \
            if cache else set()

    return _ROW_CACHE


def search_table_rows(question, limit=10):
    terms = query_tokens(question)

    if not terms:
        return []

    ranked = []
    rows = load_rows()

    for row in rows:
        row_tokens = row["_tokens"]
        matched = [term for term in terms if term in row_tokens]
        fuzzy = []

        # Tolerate small spelling differences in names.
        if process is not None:
            for term in terms:
                if (
                    term not in row_tokens
                    and term.isalpha()
                    and len(term) >= 4
                    and row["_words"]
                    and process.extractOne(
                        term,
                        row["_words"],
                        scorer=fuzz.ratio,
                        score_cutoff=FUZZY_CUTOFF,
                    )
                ):
                    fuzzy.append(term)

        if not matched and not fuzzy:
            continue

        value_text = " ".join(row["values"]).lower()
        phrase_bonus = (
            2 if len(terms) >= 2
            and " ".join(terms) in value_text
            else 0
        )

        item = {
            key: value for key, value in row.items()
            if not key.startswith("_")
        }
        item.update(
            matched_terms=matched + fuzzy,
            score=len(set(matched)) + 0.8 * len(set(fuzzy)) + phrase_bonus,
        )
        ranked.append(item)

    ranked.sort(
        key=lambda item: (item["score"], -item["page"]),
        reverse=True,
    )

    # With several meaningful words, one stray match is not a real match.
    # Words that appear nowhere in the PDFs (filler like "tell") are not
    # counted when deciding how many matches a row needs.
    if len(terms) >= 2:
        known = [term for term in terms if term in _VOCAB]
        needed = max(1, min(2, len(known)))
        ranked = [
            item for item in ranked
            if len(set(item["matched_terms"])) >= needed
        ]

    return ranked[:max(1, int(limit))]


def table_text_blocks(max_chars=1200):
    """
    Render every table row as 'Header: value | Header: value' text, grouped
    into passage-sized blocks, so table contents can also be found (and
    shown to the LLM) by the normal PDF text search.
    """
    blocks = []
    current = None

    def flush():
        if current and current["lines"]:
            blocks.append({
                "document": current["filename"],
                "page": current["page"],
                "text": "\n".join(current["lines"]),
                "first_value": current["first_value"],
            })

    for item in load_rows():
        fields = [
            (
                clean_cell(value)
                if re.fullmatch(r"Column \d+", clean_cell(header))
                else f"{clean_cell(header) or 'Value'}: {clean_cell(value)}"
            )
            for header, value in zip(item["headers"], item["values"])
            if clean_cell(value)
        ]

        if not fields:
            continue

        line = " | ".join(fields)
        key = (item["filename"], item["page"], item["table_number"])

        if (
            current is None
            or current["key"] != key
            or sum(len(x) + 1 for x in current["lines"]) + len(line)
            > max_chars
        ):
            flush()
            current = {
                "key": key,
                "filename": item["filename"],
                "page": item["page"],
                "lines": [],
                "first_value": next(
                    (clean_cell(v) for v in item["values"] if clean_cell(v)),
                    "",
                ),
            }

            if item["context"]:
                current["lines"].append(f"Table heading: {item['context']}")

        current["lines"].append(line)

    flush()
    return blocks


def format_table_row(item):
    fields = []

    for header, value in zip(
        item.get("headers", []),
        item.get("values", []),
    ):
        header = clean_cell(header) or "Value"
        value = clean_cell(value)

        if value:
            fields.append(f"{header}: {value}")

    return {
        "filename": item["filename"],
        "page": item["page"],
        "table_number": item["table_number"],
        "row_number": item["row_number"],
        "context": item.get("context", ""),
        "fields": fields,
    }


POLICY_TOPIC_PATTERN = re.compile(
    r"\b(allowances?|overtime|leaves?|holidays?|transfer|"
    r"disturbance|reimbursement|reimburse|foreign travel|"
    r"daily allowance|food allowance|meal allowance|"
    r"mobile phone allowance|mobile phone|probation|office hours|"
    r"working hours?|attendance|casual|sick|privilege|maternity|"
    r"gratuity|food|meal|lunch|dinner)\b",
    re.IGNORECASE,
)


def is_policy_question(question):
    """Return True for employee-policy questions, not record lookups."""
    return bool(POLICY_TOPIC_PATTERN.search(clean_cell(question)))


def is_direct_table_lookup(question, matches):
    """
    A direct lookup is a question about one specific record (a person, an
    extension, a grade). Decided by how well the best row matches the
    question, not by trigger words alone.
    """
    # Policy questions must use PDF passage retrieval. Words such as
    # "mobile", "phone" and "allowance" can otherwise make an employee
    # directory row look like a direct lookup and suppress the policy text.
    if is_policy_question(question):
        return False

    if not matches:
        return False

    q = question.lower().strip()

    if re.match(r"^(explain|describe|why|summari[sz]e|how)\b", q):
        return False

    terms = query_tokens(question)
    matched = set(matches[0].get("matched_terms", []))

    if not terms or not matched:
        return False

    coverage = len(matched) / len(terms)

    field_terms = (
        r"\b(extension|ext\.?|extn|phone|telephone|mobile|region|"
        r"designation|grade|price|stock|quantity|part number|serial|"
        r"email|contact|department|office)\b"
    )

    # Asking for a specific field of a record: most terms must hit one row.
    if re.search(field_terms, q):
        return coverage >= 0.75

    # Otherwise require a clearly specific match (e.g. a full name).
    return len(matched) >= 2 and coverage >= 0.8


if __name__ == "__main__":
    rebuild_table_index()

    print("\nTable search test. Type 'exit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        results = search_table_rows(question)

        if not results:
            print("No matching table rows found.")
            continue

        for item in results:
            formatted = format_table_row(item)

            print(
                f"\nSource: {formatted['filename']} "
                f"| Page: {formatted['page']}"
            )

            if formatted["context"]:
                print(f"Context: {formatted['context']}")

            print("\n".join(formatted["fields"]))