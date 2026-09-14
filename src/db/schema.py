import sqlite3
from contextlib import contextmanager

from config.settings import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    page_id INTEGER PRIMARY KEY,
    title TEXT UNIQUE NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('controversial', 'control'))
);

CREATE TABLE IF NOT EXISTS revisions (
    rev_id INTEGER PRIMARY KEY,
    page_id INTEGER NOT NULL REFERENCES pages(page_id),
    parent_id INTEGER,
    user TEXT,
    user_id INTEGER,
    anon INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL,
    size INTEGER,
    size_diff INTEGER,
    comment TEXT,
    is_minor INTEGER DEFAULT 0,
    sha1 TEXT,
    is_revert INTEGER DEFAULT NULL,
    reverted_to_rev_id INTEGER DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_revisions_page_id ON revisions(page_id);
CREATE INDEX IF NOT EXISTS idx_revisions_user ON revisions(user);
CREATE INDEX IF NOT EXISTS idx_revisions_timestamp ON revisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_revisions_sha1 ON revisions(sha1);
"""


@contextmanager
def get_connection(db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    init_db()
    print(f"Veritabanı hazır: {DB_PATH}")
