import re

from config import OLLAMA_MODEL, PDF_ROUTE_SIMILARITY
from document_qa import answer_pdf_question
from document_search import search_documents, stem
from llm import answer_question as answer_website_question
from llm import chat_with_model


def _stems(words):
    return {stem(word) for word in words}


# Topics that live in the internal PDFs (policies, HR rules, directory).
PDF_FIRST_TERMS = _stems([
    "extension", "ext", "extn", "telephone", "phone", "employee",
    "designation", "allowance", "assignment", "policy", "policies",
    "directory", "staff", "residence", "mobile", "overtime", "department",
    "leave", "claim", "reimbursement", "grade", "holiday", "attendance",
    "casual", "sick", "privilege", "maternity", "lwp", "gratuity",
    "salary", "probation", "transfer", "disturbance", "baggage", "tour",
    "lunch", "food", "meal", "document", "pdf", "procedure", "handbook",
    "entitled", "entitlement", "benefit", "aml", "arm", "ecom",
])

# Topics that live on the website. "aesl" is deliberately not listed: it's
# the company name and appears in questions of every kind.
WEBSITE_FIRST_TERMS = _stems([
    "website", "product", "service", "machine", "engine", "part", "dealer",
    "catalogue", "catalog", "warranty", "rental", "equipment", "compressor",
    "generator", "caterpillar", "excavator", "loader",
])

# Words that suggest a question depends on the previous turn.
FOLLOW_UP_HINTS = {
    "he", "she", "his", "her", "him", "it", "its", "they", "their",
    "them", "that", "this", "those", "these", "there", "also", "same",
}

MAX_HISTORY_TURNS = 6


def normalize_result(result):
    """Keep a consistent response format for the frontend."""
    if isinstance(result, dict):
        normalized = {
            "answer": str(result.get("answer", "")).strip(),
            "sources": result.get("sources", []),
        }

        if "found" in result:
            normalized["found"] = bool(result["found"])

        if result.get("llm_error"):
            normalized["llm_error"] = True

        return normalized

    return {"answer": str(result).strip(), "sources": []}


def looks_unanswered(result):
    """Detect a clear no-answer response so another source can be tried."""
    answer = result.get("answer", "").lower().strip()

    if not answer:
        return True

    # Answer functions report whether they actually found anything.
    if result.get("found") is False or "not_found" in answer:
        return True

    no_answer_phrases = (
        "couldn't find a matching record",
        "could not find a matching record",
        "couldn't find relevant information",
        "could not find relevant information",
        "couldn't confirm",
        "could not confirm",
        "information is not available",
        "i don't have enough information",
        "i do not have enough information",
        "i couldn't find",
        "i could not find",
        "no matching records found",
        "no matching table rows found",
        "does not include",
        "does not specify",
        "does not mention",
        "does not contain",
        "does not provide",
        "doesn't include",
        "doesn't specify",
        "doesn't mention",
        "doesn't contain",
        "doesn't provide",
        "not mentioned",
        "not specified",
        "no information",
        "no details",
        "cannot find",
        "can't find",
        "does not list",
        "doesn't list",
        "not listed",
        "no mention",
        "does not appear",
        "there is no information",
    )

    # Only short replies count as "no answer"; a long answer that happens
    # to mention a limitation still contains useful information.
    if len(answer) < 350 and any(p in answer for p in no_answer_phrases):
        return True

    return False


def rewrite_follow_up(question, history):
    """
    Turn a follow-up like "what about his region?" into a standalone
    question using the recent conversation. Returns the original question
    if no rewrite is needed or the rewrite looks unreliable.
    """
    if not history:
        return question

    lowered = question.lower().strip()
    word_list = re.findall(r"[a-z0-9']+", lowered)
    words = set(word_list)
    content = [
        w for w in word_list
        if w not in {"what", "about", "how", "and", "also", "the", "a", "an",
                     "of", "for", "is", "are"}
    ]

    # Only rewrite when the question really leans on the previous turn:
    # a pronoun ("his region"), or a bare "and grade 5?" / "what about X?"
    # with almost nothing else in it. "What about mobile phone allowance"
    # is a new, complete topic and is left alone.
    leans_on_history = bool(words & FOLLOW_UP_HINTS) or (
        lowered.startswith(("and ", "also ", "what about", "how about"))
        and len(content) <= 2
    )

    if not leans_on_history:
        return question

    transcript = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content'][:400]}"
        for turn in history[-MAX_HISTORY_TURNS:]
    )

    try:
        rewritten = chat_with_model(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest message as one "
                        "standalone question, replacing pronouns and "
                        "references with the names or topics from the "
                        "conversation. If it is already standalone, "
                        "repeat it unchanged. Output only the question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation:\n{transcript}\n\n"
                        f"Latest message: {question}"
                    ),
                },
            ],
            temperature=0.0,
        )
    except Exception as exc:
        print(f"Follow-up rewrite failed: {exc}")
        return question

    rewritten = rewritten.strip().strip('"')

    if not rewritten or len(rewritten) > 300 or "\n" in rewritten:
        return question

    print(f"[follow-up] {question!r} -> {rewritten!r}")
    return rewritten


def clean_history(history):
    """Accept only simple {role, content} turns from the client."""
    cleaned = []

    if not isinstance(history, list):
        return cleaned

    for turn in history[-MAX_HISTORY_TURNS:]:
        if not isinstance(turn, dict):
            continue

        role = turn.get("role")
        content = str(turn.get("content", "")).strip()

        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:1000]})

    return cleaned


def pdf_looks_relevant(question):
    """
    For questions that name neither a PDF topic nor a website topic, check
    whether the PDFs hold a passage close in meaning to the question.
    """
    try:
        results = search_documents(question, limit=3)
    except Exception as exc:
        print(f"PDF relevance check failed: {exc}")
        return False

    return any(
        (result.get("similarity") or 0) >= PDF_ROUTE_SIMILARITY
        or result.get("strong")
        for result in results
    )


def answer_hybrid_question(question, history=None):
    question = (question or "").strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    question = rewrite_follow_up(question, clean_history(history))

    stems = {stem(w) for w in re.findall(r"[a-z0-9]+", question.lower())}

    pdf_topic = bool(stems & PDF_FIRST_TERMS)
    website_topic = bool(stems & WEBSITE_FIRST_TERMS)

    # Employee, directory, and internal policy questions normally need the
    # local PDF records first; website/product questions the scraped website.
    # When the wording doesn't settle it (neither topic, or both), see
    # whether the PDFs really have something on the question.
    if pdf_topic != website_topic:
        pdf_first = pdf_topic
    else:
        pdf_first = pdf_looks_relevant(question)

    if pdf_first:
        first = answer_pdf_question
        second = answer_website_question
    else:
        first = answer_website_question
        second = answer_pdf_question

    try:
        primary = normalize_result(first(question))
    except Exception as exc:
        print(f"Primary source search failed: {exc}")
        primary = {"answer": "", "sources": []}

    if not looks_unanswered(primary):
        return {"answer": primary["answer"], "sources": primary["sources"]}

    # If the language model itself is down, the other source would fail the
    # same way (and wait for another timeout). Say what is wrong instead.
    if primary.get("llm_error"):
        return {"answer": primary["answer"], "sources": []}

    try:
        secondary = normalize_result(second(question))
    except Exception as exc:
        print(f"Fallback source search failed: {exc}")
        secondary = {"answer": "", "sources": []}

    if not looks_unanswered(secondary):
        return {
            "answer": secondary["answer"],
            "sources": secondary["sources"],
        }

    # Neither source provided a satisfactory answer.
    # Avoid inventing company information.
    return {
        "answer": (
            "I couldn't find a reliable answer in the available AESL "
            "website content or local PDF documents. Could you rephrase "
            "your question or provide another detail?"
        ),
        "sources": (
            primary.get("sources", [])
            + secondary.get("sources", [])
        ),
    }