"""
Faz 5 ana script: her revizyonun edit summary'sine sentiment ve çatışma
anahtar kelime skoru ekler, sonucu `revisions_nlp` tablosuna yazar; ayrıca
tartışmalı vs kontrol sayfalarında TF-IDF ile öne çıkan kelimeleri karşılaştırır.

Kullanım:
    python -m src.phase5_nlp.run_phase5
"""

import sqlite3

import pandas as pd

from config.settings import DB_PATH
from src.phase5_nlp.sentiment import (
    count_conflict_keywords,
    score_sentiment,
    strip_tool_boilerplate,
)
from src.phase5_nlp.tfidf_features import fit_tfidf, top_terms_by_group


def load_revisions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT rev_id, page_title, page_label, comment FROM revisions_clean", conn
        )


def main():
    print("1/3 Revizyonlar okunuyor...")
    df = load_revisions()
    df["comment"] = df["comment"].fillna("")
    print(f"     {len(df)} revizyon yüklendi")

    print("2/3 Araç şablonları temizlenip sentiment/çatışma kelime skorları hesaplanıyor...")
    df["human_comment"] = df["comment"].apply(strip_tool_boilerplate)
    # Sentiment ve anahtar kelime skorunu TEMİZLENMİŞ metin üzerinde hesaplıyoruz
    # ki "Special:Contributions" gibi şablon kelimeler skoru bozmasın.
    df["sentiment_score"] = df["human_comment"].apply(score_sentiment)
    df["conflict_keyword_count"] = df["human_comment"].apply(count_conflict_keywords)
    df["comment_length"] = df["comment"].str.len()
    df["has_comment"] = (df["comment"].str.strip() != "").astype(int)
    # Yorum tamamen araç şablonuysa (temizledikten sonra hiçbir şey kalmadıysa)
    # bu, editörün ekstra bir açıklama YAZMADIĞI anlamına gelir — kendi başına
    # bir sinyal (örn. sıradan vandalizm temizliğinde ek açıklamaya gerek duyulmaz).
    df["is_auto_generated_summary"] = (
        (df["has_comment"] == 1) & (df["human_comment"].str.strip() == "")
    ).astype(int)

    nlp_features = df[
        [
            "rev_id",
            "sentiment_score",
            "conflict_keyword_count",
            "comment_length",
            "has_comment",
            "is_auto_generated_summary",
        ]
    ]
    with sqlite3.connect(DB_PATH) as conn:
        nlp_features.to_sql("revisions_nlp", conn, if_exists="replace", index=False)
    print(f"     revisions_nlp: {len(nlp_features)} satır yazıldı")

    print("\n--- Grup karşılaştırması: sentiment ve çatışma kelimesi ortalamaları ---")
    print(
        df.groupby("page_label")[
            ["sentiment_score", "conflict_keyword_count", "has_comment", "is_auto_generated_summary"]
        ].mean()
    )

    print("3/3 TF-IDF eğitiliyor (temizlenmiş insan yorumları üzerinde)...")
    non_empty = df[df["human_comment"].str.strip() != ""]
    print(f"     {len(non_empty)}/{len(df)} revizyonda araç şablonu dışında gerçek metin var")
    vectorizer, matrix = fit_tfidf(non_empty["human_comment"], max_features=500)

    for label in ["controversial", "control"]:
        print(f"\n--- '{label}' sayfalarında en öne çıkan 15 kelime (TF-IDF) ---")
        top = top_terms_by_group(non_empty["page_label"].values, label, vectorizer, matrix, top_n=15)
        print(", ".join(f"{term}({score:.3f})" for term, score in top))


if __name__ == "__main__":
    main()
