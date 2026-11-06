# AESL Assistant

Local chatbot that answers from the AESL website and internal PDF documents
using a local Ollama model. Nothing leaves your machine.

## Setup

1. Install [Ollama](https://ollama.com) and download the model:

       ollama pull qwen2.5:7b

   (Needs roughly 8 GB of free RAM. On a weaker machine, set
   `OLLAMA_MODEL = "qwen2.5:3b"` in `config.py`.)

2. Install the Python packages:

       pip install -r requirements.txt

3. Put your PDFs in the `documents/` folder.

4. Crawl the website once (re-run whenever you want fresh website content):

       python scraper.py

5. Start the assistant:

       python app.py

   then open http://127.0.0.1:5000

The PDF index is rebuilt automatically at startup whenever the files in
`documents/` change. Scanned PDFs are OCR'd only if Tesseract is installed;
otherwise a warning is printed for each page without text.

## Testing pieces on their own

| Command                       | What it checks                          |
|-------------------------------|-----------------------------------------|
| `python document_reader.py`   | PDF text extraction                     |
| `python document_search.py`   | PDF passage search (rebuilds the index) |
| `python document_tables.py`   | Table / directory row search            |
| `python document_qa.py`       | Full PDF question answering             |
| `python search.py`            | Website search                          |
| `python llm.py`               | Full website question answering         |

## Files

- `config.py` - all settings (model, context size, crawl limits)
- `scraper.py`, `database.py`, `search.py`, `llm.py` - website crawling, storage, search, answers
- `document_reader.py`, `document_tables.py`, `document_search.py`, `document_qa.py` - PDF extraction, tables, passage search, answers
- `hybrid_qa.py` - chooses between PDFs and website, handles follow-up questions
- `app.py`, `templates/index.html` - web server and chat page
