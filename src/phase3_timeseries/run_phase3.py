"""
Faz 3 ana script: birkaç örnek sayfa için zaman serisi çizgi grafiği ve
takvim heatmap'i üretip reports/figures/ altına kaydeder.

Kullanım:
    python -m src.phase3_timeseries.run_phase3
"""

import sqlite3

import pandas as pd

from config.settings import DB_PATH, PROJECT_ROOT
from src.phase3_timeseries.analysis import plot_calendar_heatmap, plot_intensity_timeline

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Faz 2'deki bulguya göre seçilmiş örnekler:
# - "gerçek edit war" imzası taşıyan (yüksek reciprocal_revert_pairs) tartışmalı sayfalar
# - kontrol grubundan iki örnek: biri tipik "vandalizm çürüme" deseni, biri ilginç istisna
EXAMPLE_PAGES = [
    ("Abortion", "controversial — en yüksek karşılıklı revert"),
    ("Taiwan", "controversial — yüksek karşılıklı revert"),
    ("Kurdistan", "controversial — yüksek karşılıklı revert"),
    ("Igneous rock", "control — yüksek revert ama dağınık (vandalizm çürümesi)"),
    ("Espresso", "control — ilginç istisna, kontrol grubunda yüksek karşılıklı revert"),
]


def load_page_revisions(title):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM revisions_clean WHERE page_title = ?", conn, params=(title,)
        )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for title, description in EXAMPLE_PAGES:
        df = load_page_revisions(title)
        if df.empty:
            print(f"  ! '{title}' için veri bulunamadı, atlanıyor")
            continue

        safe_name = title.lower().replace(" ", "_")
        timeline_path = FIGURES_DIR / f"timeline_{safe_name}.png"
        heatmap_path = FIGURES_DIR / f"calendar_{safe_name}.png"

        plot_intensity_timeline(df, title, timeline_path, freq="W")
        plot_calendar_heatmap(df, title, heatmap_path)
        print(f"{title} ({description}): {len(df)} revizyon -> {timeline_path.name}, {heatmap_path.name}")

    with sqlite3.connect(DB_PATH) as conn:
        scores = pd.read_sql_query("SELECT * FROM page_conflict_scores", conn)

    print("\n--- Grup karşılaştırması: ortalama edit_burst_score ---")
    print(scores.groupby("page_label")["edit_burst_score"].mean())

    print("\n--- Grup karşılaştırması: ortalama reciprocal_revert_pairs ---")
    print(scores.groupby("page_label")["reciprocal_revert_pairs"].mean())


if __name__ == "__main__":
    main()
