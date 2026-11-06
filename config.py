from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "website.db"

START_URL = "https://aesl.com.pk/"
ALLOWED_HOST = "aesl.com.pk"

# Raise this after reviewing initial crawl results.
MAX_PAGES = 300
REQUEST_TIMEOUT = 25
REQUEST_DELAY_SECONDS = 1.0

USER_AGENT = "AESLWebsiteAssistant/1.0 (local website indexer)"

# Local model used for both website and PDF answers.
# Install it once with:  ollama pull qwen2.5:7b
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_TIMEOUT_SECONDS = 240

# Ollama silently truncates prompts longer than num_ctx, so keep this
# comfortably above MAX_CONTEXT_CHARS (10,000 chars is roughly 3k tokens).
OLLAMA_NUM_CTX = 8192

# Keep the model loaded between questions so replies start quickly.
OLLAMA_KEEP_ALIVE = "30m"

# Meaning-based PDF search (install once with: ollama pull nomic-embed-text).
# Similarity is 0-1; passages below EMBED_MIN_SIMILARITY are ignored, and a
# match at or above PDF_ROUTE_SIMILARITY sends an unclear question to the
# PDFs first. Tune these if answers pick up irrelevant or miss relevant text.
EMBED_MODEL = "nomic-embed-text"
EMBED_MIN_SIMILARITY = 0.55
PDF_ROUTE_SIMILARITY = 0.60

# Words people use that the documents may word differently. Used by the
# keyword search (meaning-based search covers more). Add your own: a question
# word maps to the document words it should also look for.
SEARCH_SYNONYMS = {
    "weekend": ["saturday", "sunday"],
    "weekends": ["saturday", "sunday"],
    "off": ["leave"],
    "vacation": ["leave", "privilege"],
    "pay": ["salary", "wages", "payment", "paid"],
    "salary": ["pay", "wages"],
    "wages": ["salary", "pay"],
    "extra": ["overtime", "additional"],
    "abroad": ["foreign", "overseas"],
    "overseas": ["foreign"],
    "family": ["spouse", "children", "dependent", "dependents"],
    "wife": ["spouse"],
    "husband": ["spouse"],
    "kids": ["children"],
    "meal": ["food", "lunch", "dinner"],
    "meals": ["food", "lunch", "dinner"],
    "food": ["meal", "lunch", "dinner"],
    "lunch": ["food", "meal"],
    "phone": ["mobile", "telephone"],
    "mobile": ["phone", "telephone"],
    "cell": ["mobile", "phone"],
    "hotel": ["lodging", "accommodation"],
    "accommodation": ["hotel", "lodging"],
    "timing": ["hours", "time"],
    "timings": ["hours", "time"],
    "shift": ["hours"],
    "sick": ["medical", "illness"],
    "illness": ["sick", "medical"],
    "pregnant": ["maternity"],
    "pregnancy": ["maternity"],
    "baby": ["maternity", "child"],
    "childbirth": ["maternity", "confinement"],
    "boss": ["manager", "head"],
    "trip": ["travel", "tour"],
    "ticket": ["passage", "fare"],
    "flight": ["passage", "airline"],
    "luggage": ["baggage"],
    "relocation": ["transfer"],
}

SEARCH_RESULTS = 5
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 200
MAX_CONTEXT_CHARS = 10000
MAX_QUESTION_LENGTH = 2000