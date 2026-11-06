import re

from config import OLLAMA_MODEL, PDF_ROUTE_SIMILARITY
from document_reader import looks_like_heading
from document_search import (
    document_chunks,
    document_mentions_grades,
    ensure_pdf_index,
    get_pdf_last_modified,
    match_document,
    search_documents,
)
from document_tables import (
    clean_cell,
    is_direct_table_lookup,
    is_policy_question,
    search_table_rows,
)
from llm import chat_with_model

MAX_RESULTS = 5
# "Summary of the X document": give the model the document itself, in order.
MAX_DOCUMENT_CHARS = 10000
SUMMARY_REQUEST = re.compile(
    r"\b(summar\w*|overview|in detail|detailed|everything|entire|whole|"
    r"full details|all details|complete)\b",
    re.I,
)
# Questions about types, differences or lists need to see more sections.
BROAD_RESULTS = 9
BROAD_QUESTION = re.compile(
    r"\b(types?|kinds?|difference|differences|different|compare|comparison|"
    r"list|all|every|overview|summar\w+|explain)\b",
    re.I,
)

NOT_FOUND_MESSAGE = (
    "I couldn't find a reliable answer in the available PDF documents. "
    "Could you rephrase your question or provide another detail?"
)


def make_pdf_source(filename, page):
    """Build a PDF citation, including its printed revision date when known."""
    source = {"filename": filename, "page": page}
    last_modified = get_pdf_last_modified(filename)

    if last_modified:
        source["last_modified"] = last_modified

    return source

# Telephone-directory style fields: the words people use in questions, and
# the keywords that identify the matching column header in the PDF
# (e.g. "Ext Direct", "Extn.", "Residence").
DIRECTORY_FIELDS = {
    "Extension": ("extension", "ext", "extn"),
    "Phone": ("phone", "telephone", "mobile", "cell"),
    "Region": ("region", "city", "location", "branch"),
    "Designation": ("designation", "title", "position", "post"),
    "Department": ("department", "dept"),
    "Email": ("email", "mail"),
}

FIELD_HEADER_KEYWORDS = {
    "Extension": ("ext",),
    "Phone": ("phone", "mobile", "cell", "tel", "residence"),
    "Region": ("region", "city", "location", "branch", "station", "area"),
    "Designation": ("designation", "title", "position"),
    "Department": ("dept", "department"),
    "Email": ("mail",),
}

SYSTEM_PROMPT = """
You are AESL's internal PDF assistant.

Answer the user's question conversationally using ONLY the supplied PDF
records and passages.

Rules:
- Answer the specific question first, then add only the detail that helps.
- Copy names, extensions, phone numbers, amounts, dates and conditions
  exactly as written. Never change or round them.
- Every passage belongs to a section (e.g. "Daily Allowance", "Food
  Allowance", "Overtime"). Use only the section the question is about.
  Never take amounts from another section or another table.
- A grade range such as "4 To 9" or "4-9" includes every grade in between.
  On rate rows like "1 To 2   Rs.200" the first range is the employee
  GRADE, never hours or days. When a rate table is asked about, list every
  row of it (including rows that say "Actual").
  If a GRADE CHECK is supplied, it was worked out by code: rely on it.
- If a section lists specific cases (for example the grades that are
  entitled to something) and the case asked about is not listed, say so
  and give the related rules the section does state.
- Include important conditions (approval needed, who it applies to,
  limits, exceptions) in plain language.
- When the document works something out and states a "Total" or final
  result, report that final figure (and briefly how it is reached), not an
  intermediate line. You may convert units (e.g. 2400 minutes = 40 hours).
- Different companies or groups can have different rules (for example
  "AESL / ARM / ECOM" versus "AML"). Give the rule for the one asked
  about, or for each if the question doesn't say.
- If the question assumes something the documents don't state (for example
  a rate "per hour"), say so, then give what the documents do state.
- For questions about types, kinds or differences, cover every type found
  in the sources, one short line each.
- A list belongs to the heading above it (the [Section: ...] label shows
  which). Never move names or items from one list to another, and never say
  a list is "not specified" when names appear under that heading.
- When asked to summarise a document, go through every section in order:
  what the document is for, its rules, then each group or team with all of
  its members, keeping names exactly as written.
- Never combine values from different records. Never invent a missing
  number, designation, region or rule.
- Reply with exactly NOT_FOUND only if the sources are about a different
  topic from the question.
- Do not mention internal search processes or "the sources".
- Reply in the same language as the question. Keep answers concise unless
  the user asks for details.

Examples (the values here are made up, only the style matters):

Sources: Passage [Section: Meals]: Staff on grade X may claim a meal
allowance of 500 per day when travelling more than 50 km from their base.
Question: Can a grade X employee claim meals for a 20 km trip?
Answer: No. The meal allowance for grade X applies only when travelling more
than 50 km from the base, so a 20 km trip doesn't qualify.

Sources: Passage [Section: Bonus]: Employees in grades 1, 2, 4 and 5 receive
an annual bonus.
GRADE CHECK for grade 3: "...grades 1, 2, 4 and 5 receive an annual bonus."
-> grade 3 is NOT in this list.
Question: Does a grade 3 employee get the annual bonus?
Answer: Grade 3 isn't among the grades listed for the annual bonus (grades 1,
2, 4 and 5).
"""


def row_fields(item):
    """Header -> value mapping for one table row."""
    return {
        clean_cell(header): clean_cell(value)
        for header, value in zip(item["headers"], item["values"])
        if clean_cell(header)
    }


def field_columns(record, field):
    """(header, value) pairs for the columns that hold this kind of field."""
    keywords = FIELD_HEADER_KEYWORDS[field]

    return [
        (header, value)
        for header, value in record.items()
        if any(keyword in header.lower() for keyword in keywords)
    ]


def directory_answer(question, table_results):
    """
    Answer employee-directory lookups straight from the matched records.
    Exact values matter here, so the LLM is not involved: it can only
    add the risk of a garbled number. Returns None when the matched rows
    aren't directory-style records.
    """
    top_score = table_results[0]["score"]
    matches = [
        item for item in table_results
        if item["score"] >= top_score - 0.01
    ]
    records = [row_fields(item) for item in matches]

    if not all(
        record.get("Name")
        and any(field_columns(record, field) for field in DIRECTORY_FIELDS)
        for record in records
    ):
        return None

    # The same record can be indexed more than once (e.g. repeated pages).
    unique = {}

    for item, record in zip(matches, records):
        unique.setdefault(tuple(sorted(record.items())), (item, record))

    matches = [item for item, _ in unique.values()]
    records = [record for _, record in unique.values()]

    q = question.lower()
    words = set(re.findall(r"[a-z]+", q))
    numbers = re.findall(r"\d{2,}", q)

    requested = [
        field for field, aliases in DIRECTORY_FIELDS.items()
        if words & set(aliases)
    ]

    # "Who has extension 339?": keep only people whose extension column
    # really holds that number (not any other column that contains it).
    if "Extension" in requested and numbers:
        narrowed = [
            (item, record)
            for item, record in zip(matches, records)
            if any(
                number in re.findall(r"\d+", value)
                for _, value in field_columns(record, "Extension")
                for number in numbers
            )
        ]

        if narrowed:
            matches = [item for item, _ in narrowed]
            records = [record for _, record in narrowed]

    if requested:
        show = list(requested)

        # Looking someone up by number or "who" also wants their role.
        if (re.match(r"^\s*who(se)?\b", q) or numbers) \
                and "Designation" not in show:
            show.insert(0, "Designation")

        # Several people share the answer: add details that tell them apart.
        if len(records) > 1:
            for field in ("Designation", "Region", "Department"):
                if field not in show and any(
                    value
                    for record in records
                    for _, value in field_columns(record, field)
                ):
                    show.append(field)
    else:
        show = None  # "Who is X?" -> every available field

    lines = []

    for record in records:
        if show is None:
            parts = [
                f"{field}: {value}"
                for field, value in record.items()
                if value and field != "Name"
            ]
        else:
            parts = []

            for field in show:
                filled = [
                    f"{header}: {value}"
                    for header, value in field_columns(record, field)
                    if value
                ]
                parts.extend(filled or [f"{field}: not listed"])

        line = record["Name"]

        if parts:
            line += " - " + ", ".join(parts)

        if line not in lines:
            lines.append(line)

    if len(lines) == 1:
        answer = lines[0] + "."
    else:
        answer = (
            f"I found {len(lines)} matching records:\n"
            + "\n".join(f"- {line}" for line in lines)
        )

    sources = []

    for item in matches:
        source = make_pdf_source(item["filename"], item["page"])

        if source not in sources:
            sources.append(source)

    return {"answer": answer, "sources": sources, "found": True}


# Words that say nothing about WHICH section or table a question is about.
GENERIC_WORDS = {
    "what", "is", "are", "the", "for", "of", "a", "an", "to", "how", "much",
    "many", "do", "does", "i", "my", "me", "we", "get", "and", "in", "on",
    "per", "grade", "grades", "allowance", "allowances", "policy", "rate",
    "rates", "details", "detail", "about", "tell", "give", "can", "will",
}


def topic_words(text):
    return set(re.findall(r"[a-z]+", text.lower())) - GENERIC_WORDS


GRADE_IN_QUESTION = re.compile(r"\bgrades?\s*[-:#]?\s*(\d{1,2})\b", re.I)
RANGE_WORDS = r"(?:to|-|\u2013|\u2014)"
NUMBER_ITEM = rf"\d{{1,2}}(?!\d)(?:\s*{RANGE_WORDS}\s*\d{{1,2}}(?!\d))?"
ROW_START = re.compile(rf"^\s*((?:{NUMBER_ITEM}[\s,]*)+)(.*)$", re.I)
SENTENCE_SPEC = re.compile(
    rf"\bgrades?\s+((?:{NUMBER_ITEM}(?:\s*(?:,|and|&)\s*|\s+(?=\d))?)+)"
    rf"(?:\s*(above|higher|over|onwards?|below|lower|under)\b)?",
    re.I,
)
AMOUNT_MARKER = re.compile(r"\b(?:rs|pkr)\b|actual|\d{3,}", re.I)


def parse_grades(spec, direction=None):
    """'1, 2, 4 and 5' / '4 To 9' / '10 11 12' -> {grades}."""
    grades = set()
    range_pattern = rf"(?<!\d)(\d{{1,2}})\s*{RANGE_WORDS}\s*(\d{{1,2}})(?!\d)"

    for low, high in re.findall(range_pattern, spec, re.I):
        if int(low) <= int(high) <= 40:
            grades.update(range(int(low), int(high) + 1))

    rest = re.sub(range_pattern, " ", spec, flags=re.I)
    grades.update(
        int(n) for n in re.findall(r"(?<!\d)\d{1,2}(?!\d)", rest)
        if int(n) <= 40
    )

    # "grade 5 and above" / "grade 9 and below"
    if grades and direction:
        if direction.lower() in ("above", "higher", "over", "onward", "onwards"):
            grades = set(range(min(grades), 41))
        else:
            grades = set(range(0, max(grades) + 1))

    return grades


def is_column_header(line):
    """An all-caps line such as 'GRADE HOTEL STAY DAILY' above a table."""
    letters = [c for c in line if c.isalpha()]

    return (
        "grade" in line.lower()
        and not re.search(r"\d", line)
        and bool(letters)
        and sum(c.isupper() for c in letters) / len(letters) > 0.7
    )


def grade_check(question, document_results):
    """
    Work out, in code, which retrieved lines apply to the grade the user
    asked about: ranges like "4 To 9" and lists like "1, 2, 3, 4, 5, 7, 8, 9"
    are easy for a small model to get wrong ("is grade 6 in that list?").
    """
    found = GRADE_IN_QUESTION.findall(question)

    if not found:
        return ""

    grade = int(found[0])
    applies, excluded = [], []
    seen = set()

    for result in document_results:
        section = ""
        header = ""

        lines = [ln.strip() for ln in result.get("text", "").splitlines()]
        lines = [ln for ln in lines if ln]

        for index, line in enumerate(lines):
            if line.startswith("[Section:"):
                section = line[len("[Section:"):].rstrip("]").strip()
                continue

            if line.startswith("Table heading:"):
                section = line[len("Table heading:"):].split(" - ")[0].strip()
                continue

            # A real heading, not a column note like "NOT EXCEEDING" that
            # sits above the first row of a rate table.
            following = lines[index + 1] if index + 1 < len(lines) else ""

            if looks_like_heading(line) and not re.match(
                r"^\d{1,2}\s*(?:to\b|-|\u2013|\d|rs\b)|^rs\b",
                following, re.I,
            ):
                section, header = line, ""
                continue

            if looks_like_heading(line):
                continue

            if is_column_header(line):
                header = line
                continue

            line = re.sub(r"\s*\|\s*", " ", line)
            key = re.sub(r"\s+", " ", line).lower()

            if key in seen:
                continue

            seen.add(key)
            label = f"[{section}] " if section else ""
            relevance = len(
                (topic_words(section) | topic_words(header))
                & topic_words(question)
            )

            # A table row: "4 To 9   Rs.8,000   Rs.4,000"
            row = ROW_START.match(line)

            if row and AMOUNT_MARKER.search(row.group(2)):
                if grade in parse_grades(row.group(1)):
                    columns = f' (columns: "{header}")' if header else ""
                    applies.append((
                        relevance,
                        f'- {label}"{line}"{columns} -> includes grade {grade}',
                    ))

                continue

            # A sentence: "...grades 1, 2, 3 are entitled to overtime..."
            for match in SENTENCE_SPEC.finditer(line):
                grades = parse_grades(match.group(1), match.group(2))

                if not grades:
                    continue

                if grade in grades:
                    applies.append((
                        relevance,
                        f'- {label}"{line}" -> includes grade {grade}',
                    ))
                else:
                    excluded.append((
                        relevance,
                        f'- {label}"{line}" -> grade {grade} is NOT in '
                        f"this list",
                    ))

                break

    # Only lines from the section or table the question is about ("food",
    # "hotel stay"); others that happen to include the grade just confuse.
    top = max([score for score, _ in applies + excluded], default=0)

    if top > 0:
        applies = [item for item in applies if item[0] == top]
        excluded = [item for item in excluded if item[0] == top]

    applies.sort(key=lambda item: -item[0])
    excluded.sort(key=lambda item: -item[0])
    lines = [text for _, text in applies[:6] + excluded[:4]]

    if not lines:
        return ""

    return (
        f"GRADE CHECK for grade {grade} (worked out by code):\n"
        + "\n".join(lines)
    )


def _numbers_in(text):
    values = set()

    for match in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            values.add(float(match.replace(",", "")))
        except ValueError:
            pass

    return values


def rate_rows(question, document_results):
    """
    Rate tables list a grade range then an amount ("1 To 2   Rs.200"), often
    without a header. A small model can read "1 To 2" as hours, so spell the
    first column out as the employee grade, for the section asked about.
    """
    if GRADE_IN_QUESTION.search(question):
        return ""           # grade_check already covers a specific grade

    best = {}               # row text -> (relevance, hint line)

    for result in document_results:
        if not document_mentions_grades(result.get("filename", "")):
            continue

        section, header = "", ""
        lines = [ln.strip() for ln in result.get("text", "").splitlines()]
        lines = [ln for ln in lines if ln]

        for index, line in enumerate(lines):
            if line.startswith("[Section:"):
                section = line[len("[Section:"):].rstrip("]").strip()
                continue

            if line.startswith("Table heading:"):
                section = line[len("Table heading:"):].split(" - ")[0].strip()
                continue

            following = lines[index + 1] if index + 1 < len(lines) else ""

            if looks_like_heading(line) and not re.match(
                r"^\d{1,2}\s*(?:to\b|-|\u2013|\d|rs\b)|^rs\b",
                following, re.I,
            ):
                section, header = line, ""
                continue

            if looks_like_heading(line):
                continue

            if is_column_header(line):
                header = line
                continue

            line = re.sub(r"\s*\|\s*", " ", line)
            match = ROW_START.match(line)

            if not (match and AMOUNT_MARKER.search(match.group(2))):
                continue

            relevance = len(
                (topic_words(section) | topic_words(header))
                & topic_words(question)
            )
            columns = f' (columns: "{header}")' if header else ""
            key = re.sub(r"\s+", " ", line).lower()
            hint = (
                f"- [{section or 'table'}] grades {match.group(1).strip()}: "
                f"{match.group(2).strip()}{columns}"
            )

            # The same row can appear as text and as a table passage: keep
            # the copy that knows its section.
            if key not in best or relevance > best[key][0]:
                best[key] = (relevance, hint)

    rows = list(best.values())
    top = max([score for score, _ in rows], default=0)

    if top > 0:
        rows = [item for item in rows if item[0] == top]

    if len(rows) < 2:
        return ""

    return (
        "RATE TABLE ROWS (the first number or range on each row is the "
        "employee GRADE, not hours or days; list every row):\n"
        + "\n".join(text for _, text in rows[:12])
    )


UNIT_NUMBER = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*\)?\s*"
    r"(min|mins|minutes?|hrs?|hours?)\b",
    re.I,
)


def _unit_numbers(text):
    """(minutes, hours) written with their unit, e.g. '2,650 Min', '5 hrs'."""
    minutes, hours = set(), set()

    for number, unit in UNIT_NUMBER.findall(text):
        value = float(number.replace(",", ""))
        (minutes if unit.lower().startswith("min") else hours).add(value)

    return minutes, hours


def unsupported_numbers(answer, question, context):
    """
    Figures in the answer that can't come from the sources. A figure counts
    as supported if it appears there, or is a plain unit conversion of a
    time written there (2,400 min = 40 hours; 5 hrs = 300 min), or the sum
    of minutes on neighbouring lines (2,350 min plus 300 min). Rates and
    other amounts are never added up: anything else counts as invented.
    """
    allowed = {round(v, 2) for v in _numbers_in(context + " " + question)}

    per_line = []

    for line in context.splitlines():
        minutes, hours = _unit_numbers(line)

        for value in minutes:
            allowed.add(round(value / 60, 2))      # minutes -> hours

        for value in hours:
            allowed.add(round(value * 60, 2))      # hours -> minutes

        per_line.append(sorted(minutes | {h * 60 for h in hours}))

    for i, first in enumerate(per_line):
        for second in per_line[i + 1:i + 3]:
            for a in first:
                for b in second:
                    total = a + b
                    allowed.add(round(total, 2))
                    allowed.add(round(total / 60, 2))

    unsupported = []

    for match in re.findall(r"\d[\d,]*(?:\.\d+)?", answer):
        value = float(match.replace(",", ""))

        # Ignore single digits such as list numbers.
        if len(re.sub(r"\D", "", match)) < 2:
            continue

        if value == int(value):
            supported = round(value, 2) in allowed
        else:
            supported = any(abs(value - y) <= 0.051 for y in allowed)

        if not supported:
            unsupported.append(match)

    return unsupported


def prepare_pdf_context(question):
    """
    Find what the PDFs say about the question: either a direct directory
    answer, or the text (and grade check) to give the language model.
    """
    # Structured table rows are only for lookups of one specific record
    # (an employee, an extension). Policy questions ("food allowance",
    # "leave") never use them: those rows match on loose words.
    table_results = []

    # "Summary of the Cross Functional Teams document": use all of it.
    named = match_document(question) if SUMMARY_REQUEST.search(question) else None

    if named:
        passages, truncated = document_chunks(named, MAX_DOCUMENT_CHARS)

        if passages:
            return _build_context(
                question, passages,
                note=(
                    "NOTE: the document is longer than what is shown here; "
                    "only its first part is included."
                    if truncated else ""
                ),
            )

    if not is_policy_question(question):
        try:
            table_results = search_table_rows(question, limit=20)
        except Exception as exc:
            print(f"Table search warning: {exc}")

        if is_direct_table_lookup(question, table_results) and table_results:
            direct = directory_answer(question, table_results)

            if direct:
                return {"direct": direct}

    # Everything else is answered from the passages: table contents are
    # indexed there too, with their section headings.
    limit = BROAD_RESULTS if BROAD_QUESTION.search(question) else MAX_RESULTS

    try:
        found = search_documents(question, limit=limit + 3)
    except Exception as exc:
        print(f"PDF text search warning: {exc}")
        found = []

    document_results = found[:limit]

    # For broad questions ("types of leave") very short passages (a title
    # block, a two-line table) rank high on length alone and crowd out the
    # real sections. Short lists matter for specific ones ("sports team").
    if BROAD_QUESTION.search(question):
        solid = [row for row in found if len(row["text"]) >= 100]
        document_results = (solid or found)[:limit]

    if not document_results:
        return {"empty": True}

    return _build_context(question, document_results)


def _page_set(page):
    """3 -> {3};  '2-3' -> {2, 3}."""
    text = str(page)

    if "-" in text:
        low, _, high = text.partition("-")

        if low.isdigit() and high.isdigit():
            return set(range(int(low), int(high) + 1))

    return {int(text)} if text.isdigit() else {text}


def compact_sources(sources):
    """Drop a page citation already covered by a wider one ('2' vs '2-3')."""
    kept = []

    for source in sources:
        pages = _page_set(source["page"])

        covered = any(
            other is not source
            and other["filename"] == source["filename"]
            and pages < _page_set(other["page"])
            for other in sources
        )

        if not covered:
            kept.append(source)

    return kept


def _build_context(question, document_results, note=""):
    """Turn retrieved passages into the text the model is given."""
    context_parts = []
    sources = []

    for result in document_results:
        filename = result.get("filename", "Unknown PDF")
        page = result.get("page", "?")
        text = result.get("text", "")

        context_parts.append(
            f"PDF PASSAGE\n"
            f"Source: {filename}, page {page}\n"
            f"Passage:\n{text}"
        )

        source = make_pdf_source(filename, page)

        if source not in sources:
            sources.append(source)

    sources = compact_sources(sources)

    check = grade_check(question, document_results) or rate_rows(
        question, document_results
    )

    if check:
        context_parts.append(check)

    if note:
        context_parts.append(note)

    return {
        "context": "\n\n---\n\n".join(context_parts),
        "sources": sources,
        "document_results": document_results,
    }


def answer_pdf_question(question):
    question = (question or "").strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "found": False,
        }

    prepared = prepare_pdf_context(question)

    if "direct" in prepared:
        return prepared["direct"]

    if prepared.get("empty"):
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "found": False}

    context = prepared["context"]
    sources = prepared["sources"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Sources:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Write a natural, direct answer using only these sources."
            ),
        },
    ]

    try:
        answer = chat_with_model(messages, temperature=0.0)

        if not answer:
            return {
                "answer": (
                    "I found relevant PDF information, but couldn't "
                    "generate a response. Please try rephrasing your "
                    "question."
                ),
                "sources": sources,
                "found": False,
            }

        if "NOT_FOUND" in answer.upper():
            # The documents may cover the topic without covering the exact
            # point ("public holiday on a weekend"). Then say what they do
            # state, and that the specific point isn't covered.
            related = any(
                row.get("strong")
                or row.get("matched", 0) >= 2
                or (row.get("similarity") or 0) >= PDF_ROUTE_SIMILARITY
                for row in prepared["document_results"]
            )
            rescue = ""

            if related:
                rescue = chat_with_model(
                    messages + [
                        {"role": "assistant", "content": answer},
                        {
                            "role": "user",
                            "content": (
                                "The sources do contain related rules, even "
                                "if they don't answer the exact question. In "
                                "two or three sentences, say what the "
                                "sources state about this topic, and say "
                                "plainly that they do not state the specific "
                                "point asked about. Use only the sources. "
                                "Do not guess. Do not reply NOT_FOUND."
                            ),
                        },
                    ],
                    temperature=0.0,
                )

            if not rescue or "NOT_FOUND" in rescue.upper():
                return {
                    "answer": NOT_FOUND_MESSAGE,
                    "sources": [],
                    "found": False,
                }

            answer = rescue

        # Every figure in the answer must come from the documents.
        bad_numbers = unsupported_numbers(answer, question, context)

        if bad_numbers:
            print(f"Unverified numbers in answer: {bad_numbers}")

            retry = chat_with_model(
                messages + [
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "These figures do not appear in the sources: "
                            + ", ".join(bad_numbers)
                            + ". Rewrite the answer using only figures that "
                            "appear in the sources. If you calculate a "
                            "figure (for example minutes to hours), show "
                            "the calculation. Do not use NOT_FOUND unless "
                            "the sources are about a different topic."
                        ),
                    },
                ],
                temperature=0.0,
            )

            if retry and "NOT_FOUND" not in retry.upper():
                answer = retry
                bad_numbers = unsupported_numbers(answer, question, context)

            if bad_numbers:
                answer += (
                    "\n\n(Note: I couldn't confirm "
                    + ", ".join(bad_numbers)
                    + " in the documents. Please double-check with the "
                    "source.)"
                )

        return {"answer": answer, "sources": sources, "found": True}

    except Exception as exc:
        print(f"Ollama answer generation failed: {exc}")

        return {
            "answer": (
                "I found relevant PDF information, but the language model "
                "isn't responding. Check that Ollama is running and that "
                f"'{OLLAMA_MODEL}' is installed (ollama pull {OLLAMA_MODEL})."
            ),
            "sources": sources,
            "found": False,
            "llm_error": True,
        }


if __name__ == "__main__":
    ensure_pdf_index()

    print("\nAESL PDF Question-Answering Chatbot")
    print(f"Using local model: {OLLAMA_MODEL}")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Chatbot stopped.")
            break

        result = answer_pdf_question(question)

        print(f"\nBot: {result['answer']}")

        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(
                    f"- {source['filename']} "
                    f"(Page {source['page']})"
                )

        print()