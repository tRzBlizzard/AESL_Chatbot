from pathlib import Path

from flask import Flask, jsonify, render_template, request

from config import MAX_QUESTION_LENGTH
from database import initialize_database
from document_search import ensure_pdf_index
from hybrid_qa import answer_hybrid_question

app = Flask(__name__)

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"


def get_pdf_documents():
    """Return PDF filenames currently present in the documents folder."""
    if not DOCUMENTS_DIR.is_dir():
        return []

    return sorted(
        (
            path.name
            for path in DOCUMENTS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=str.casefold,
    )


# Prepare the website database and (re)build the PDF index if the files in
# documents/ have changed since the last run.
initialize_database()
ensure_pdf_index()


@app.get("/")
def home():
    pdf_documents = get_pdf_documents()
    return render_template(
        "index.html",
        pdf_documents=pdf_documents,
        pdf_document_count=len(pdf_documents),
    )


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({
            "error": (
                "Please keep your question under "
                f"{MAX_QUESTION_LENGTH:,} characters."
            )
        }), 400

    try:
        result = answer_hybrid_question(question, data.get("history"))
    except Exception as exc:
        print(f"Chat error: {exc}")
        return jsonify({
            "error": "Something went wrong while answering. Please try again."
        }), 500

    return jsonify(result)


if __name__ == "__main__":
    # Keep the current host setting for local-only access.
    # For office-LAN access, use host="0.0.0.0" and configure firewall/authentication.
    from waitress import serve

    print("AESL assistant running at http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000, threads=4)
    # serve(app, host="0.0.0.0", port=5000, threads=4)
