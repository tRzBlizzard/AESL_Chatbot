
import re
import sqlite3
from datetime import datetime, timezone

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DATABASE_PATH,
)


def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


# Bump this whenever the search table layout changes. The search index is
# rebuilt automatically from the saved pages, so no re-crawl is needed.
SCHEMA_VERSION = "2"


def initialize_database():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        outdated = row is None or row["value"] != SCHEMA_VERSION

        if outdated:
            # Older versions used a different tokenizer and a separate
            # page_search table that the crawler never filled.
            conn.execute("DROP TABLE IF EXISTS chunk_search")
            conn.execute("DROP TABLE IF EXISTS page_search")

        # The porter tokenizer lets "product" match "products", etc.
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search
            USING fts5(
                url UNINDEXED,
                title,
                content,
                tokenize='porter unicode61'
            )
        """)

        if outdated:
            for page in conn.execute(
                "SELECT url, title, content FROM pages"
            ).fetchall():
                conn.executemany("""
                    INSERT INTO chunk_search (url, title, content)
                    VALUES (?, ?, ?)
                """, [
                    (page["url"], page["title"], chunk)
                    for chunk in split_into_chunks(page["content"])
                ])

            conn.execute("""
                INSERT OR REPLACE INTO meta (key, value)
                VALUES ('schema_version', ?)
            """, (SCHEMA_VERSION,))


def split_into_chunks(text):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    while start < length:
        end = min(start + CHUNK_SIZE, length)

        # Prefer ending at a paragraph or sentence boundary.
        if end < length:
            minimum_end = start + int(CHUNK_SIZE * 0.65)
            candidates = [
                text.rfind("\n\n", minimum_end, end),
                text.rfind(". ", minimum_end, end),
                text.rfind(" ", minimum_end, end),
            ]
            boundary = max(candidates)

            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(start + 1, end - CHUNK_OVERLAP)

    return chunks


def save_page(url, title, content):
    chunks = split_into_chunks(content)
    timestamp = datetime.now(timezone.utc).isoformat()

    with connect() as conn:
        # Update the page and all its search chunks in one transaction.
        conn.execute("""
            INSERT INTO pages (url, title, content, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                updated_at = excluded.updated_at
        """, (url, title, content, timestamp))

        conn.execute(
            "DELETE FROM chunk_search WHERE url = ?", (url,)
        )

        conn.executemany("""
            INSERT INTO chunk_search (url, title, content)
            VALUES (?, ?, ?)
        """, [(url, title, chunk) for chunk in chunks])


def database_stats():
    initialize_database()

    with connect() as conn:
        page_count = conn.execute(
            "SELECT COUNT(*) FROM pages"
        ).fetchone()[0]

        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunk_search"
        ).fetchone()[0]

        latest = conn.execute(
            "SELECT MAX(updated_at) FROM pages"
        ).fetchone()[0]

    return {
        "pages": page_count,
        "chunks": chunk_count,
        "latest_update": latest,
    }


if __name__ == "__main__":
    initialize_database()
    print("SQLite FTS5 database initialized.")
    print(database_stats())
