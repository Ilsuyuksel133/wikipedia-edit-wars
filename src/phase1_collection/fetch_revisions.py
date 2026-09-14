"""
Faz 1 ana script: config/pages.py listesindeki tüm sayfalar için revizyon
geçmişini çekip data/raw/wikipedia_edit_wars.db içine yazar.

Kullanım:
    python -m src.phase1_collection.fetch_revisions
"""

import sys
import time

from tqdm import tqdm

from config.pages import all_pages_with_labels
from config.settings import MAX_REVISIONS_PER_PAGE, REQUEST_DELAY_SECONDS
from src.db.schema import get_connection, init_db
from src.phase1_collection.wiki_api import fetch_page_revisions


def upsert_page(conn, page_id, title, label):
    conn.execute(
        "INSERT INTO pages (page_id, title, label) VALUES (?, ?, ?) "
        "ON CONFLICT(page_id) DO UPDATE SET title=excluded.title, label=excluded.label",
        (page_id, title, label),
    )


def insert_revision(conn, rev):
    conn.execute(
        """
        INSERT OR IGNORE INTO revisions
            (rev_id, page_id, parent_id, user, user_id, anon, timestamp,
             size, size_diff, comment, is_minor, sha1)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rev.get("revid"),
            rev.get("page_id"),
            rev.get("parentid"),
            rev.get("user"),
            rev.get("userid"),
            1 if "anon" in rev else 0,
            rev.get("timestamp"),
            rev.get("size"),
            None,  # size_diff: Faz 2'de önceki revizyonla karşılaştırılarak hesaplanacak
            rev.get("comment"),
            1 if "minor" in rev else 0,
            rev.get("sha1"),
        ),
    )


def collect_page(conn, title, label, max_revisions=MAX_REVISIONS_PER_PAGE):
    revisions = list(fetch_page_revisions(title, max_revisions=max_revisions))
    if not revisions:
        print(f"  ! '{title}' için revizyon bulunamadı (sayfa yok/yönlendirme olabilir)")
        return 0

    page_id = revisions[0]["page_id"]
    page_title = revisions[0]["page_title"]
    upsert_page(conn, page_id, page_title, label)

    for rev in revisions:
        insert_revision(conn, rev)

    return len(revisions)


def already_collected_titles(conn):
    """Veritabanında zaten en az bir revizyonu bulunan sayfa başlıklarını döndürür
    (script yarıda kesilip tekrar çalıştırılırsa aynı sayfayı boşuna tekrar çekmemek için)."""
    rows = conn.execute(
        "SELECT DISTINCT p.title FROM pages p JOIN revisions r ON p.page_id = r.page_id"
    ).fetchall()
    return {row[0] for row in rows}


def main():
    init_db()
    pages = all_pages_with_labels()

    with get_connection() as conn:
        done = already_collected_titles(conn)

    remaining = [(title, label) for title, label in pages if title not in done]
    if done:
        print(f"{len(done)} sayfa zaten çekilmiş, atlanıyor. {len(remaining)} sayfa kaldı.")

    total = 0
    with get_connection() as conn:
        for i, (title, label) in enumerate(tqdm(remaining, desc="Sayfalar")):
            if i > 0:
                # Sayfalar arasında da bekleyelim ki Wikipedia bizi
                # "çok hızlı istek atıyor" diye engellemesin (429 hatası).
                time.sleep(REQUEST_DELAY_SECONDS)
            try:
                count = collect_page(conn, title, label)
                total += count
                conn.commit()
            except Exception as exc:
                print(f"  ! '{title}' işlenirken hata: {exc}", file=sys.stderr)

    print(f"\nToplam {total} yeni revizyon, {len(remaining)} sayfa için çekildi.")


if __name__ == "__main__":
    main()
