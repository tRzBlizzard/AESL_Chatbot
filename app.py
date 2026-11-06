from flask import Flask, jsonify, render_template, request

from config import MAX_QUESTION_LENGTH
from database import initialize_database
from document_search import ensure_pdf_index
from hybrid_qa import answer_hybrid_question

app = Flask(__name__)

# Prepare the website database and (re)build the PDF index if the files in
# documents/ have changed since the last run.
initialize_database()
ensure_pdf_index()


@app.get("/")
def home():
    return render_template("index.html")


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
    # Local laptop only. Add authentication/HTTPS before office-network deployment.
    from waitress import serve

    print("AESL assistant running at http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000, threads=4)
