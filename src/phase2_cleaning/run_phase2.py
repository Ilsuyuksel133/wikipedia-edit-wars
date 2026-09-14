"""
Faz 2 ana script: revisions tablosunu okur, bot'ları filtreler, boyut
farklarını hesaplar, revert tespiti yapar, editör/sayfa feature'larını
üretir ve sonucu `revisions_clean` + `page_conflict_scores` tablolarına yazar.

Kullanım:
    python -m src.phase2_cleaning.run_phase2
"""

import pandas as pd

from src.db.schema import get_connection
from src.phase2_cleaning.bot_filter import filter_bot_revisions
from src.phase2_cleaning.features_editor import (
    compute_editor_features,
    compute_page_conflict_score,
)
from src.phase2_cleaning.revert_detection import detect_reverts


def load_revisions():
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT r.*, p.title AS page_title, p.label AS page_label
            FROM revisions r
            JOIN pages p ON r.page_id = p.page_id
            """,
            conn,
        )
    # Nadiren (gizlilik/oversight nedeniyle) kullanıcı adı gizlenmiş revizyonlar
    # olabilir (user=NULL) — editör istatistiklerinde ayrı, belirgin bir grup
    # olarak görünsünler diye sentinel bir değer veriyoruz.
    df["user"] = df["user"].fillna("[gizli_kullanıcı]")
    return df


def compute_size_diff(df):
    """Her revizyonun bir önceki revizyona göre boyut farkını hesaplar
    (sayfa bazında, zaman sırasına göre)."""
    df = df.sort_values(["page_id", "timestamp"]).reset_index(drop=True)
    diffs = df.groupby("page_id")["size"].diff()
    # ilk revizyonun "farkı" yoktur, sıfırdan o boyuta gelmiş sayılır
    df["size_diff"] = diffs.fillna(df["size"])
    return df


def apply_revert_detection(df):
    """detect_reverts'i her sayfa için ayrı ayrı çalıştırıp sonucu DataFrame'e ekler."""
    df = df.sort_values(["page_id", "timestamp"]).reset_index(drop=True)

    is_revert, reverted_to, is_reverted, reverted_by = [], [], [], []
    for _, group in df.groupby("page_id", sort=False):
        revs = group[["rev_id", "sha1"]].to_dict("records")
        result = detect_reverts(revs)
        for rev_id in group["rev_id"]:
            info = result[rev_id]
            is_revert.append(info["is_revert"])
            reverted_to.append(info["reverted_to_rev_id"])
            is_reverted.append(info["is_reverted"])
            reverted_by.append(info["reverted_by_rev_id"])

    df["is_revert"] = is_revert
    df["reverted_to_rev_id"] = reverted_to
    df["is_reverted"] = is_reverted
    df["reverted_by_rev_id"] = reverted_by
    return df


def main():
    print("1/5 Revizyonlar veritabanından okunuyor...")
    df = load_revisions()
    print(f"     {len(df)} revizyon yüklendi")

    print("2/5 Bot editörler filtreleniyor...")
    before = len(df)
    df = filter_bot_revisions(df)
    print(f"     {before - len(df)} bot revizyonu çıkarıldı, {len(df)} kaldı")

    print("3/5 Boyut farkları hesaplanıyor...")
    df = compute_size_diff(df)

    print("4/5 Revert tespiti yapılıyor (sayfa sayfa)...")
    df = apply_revert_detection(df)
    print(f"     {int(df['is_revert'].sum())} revert tespit edildi")

    print("5/5 Editör ve sayfa feature'ları hesaplanıyor...")
    df = compute_editor_features(df)

    scores = []
    for page_id, group in df.groupby("page_id"):
        s = compute_page_conflict_score(group)
        s["page_id"] = page_id
        s["page_title"] = group["page_title"].iloc[0]
        s["page_label"] = group["page_label"].iloc[0]
        scores.append(s)
    scores_df = pd.DataFrame(scores)

    with get_connection() as conn:
        df.to_sql("revisions_clean", conn, if_exists="replace", index=False)
        scores_df.to_sql("page_conflict_scores", conn, if_exists="replace", index=False)

    print(f"\nrevisions_clean: {len(df)} satır yazıldı")
    print(f"page_conflict_scores: {len(scores_df)} satır yazıldı")

    cols = [
        "page_title", "page_label", "revert_rate", "revert_concentration_hhi",
        "anon_reverted_ratio", "reciprocal_revert_pairs",
    ]

    print("\nEn çatışmalı 5 sayfa (ham revert oranına göre — vandalizmle karışabilir):")
    print(scores_df.sort_values("revert_rate", ascending=False).head(5)[cols].to_string(index=False))

    print("\nEn çatışmalı 5 sayfa (karşılıklı revert çifti sayısına göre — gerçek edit war sinyali):")
    print(scores_df.sort_values("reciprocal_revert_pairs", ascending=False).head(5)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
