"""
Faz 4 ana script: her sayfa için revert grafiği kurar, merkezilik ve
topluluk (Louvain) tespiti yapar. Sonuçları `editor_network_features` ve
`page_network_features` tablolarına yazar; birkaç örnek sayfa için de
grafiği görselleştirip reports/figures/ altına kaydeder.

Kullanım:
    python -m src.phase4_network.run_phase4
"""

import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from config.settings import DB_PATH, PROJECT_ROOT
from src.phase4_network.build_graph import (
    build_revert_graph,
    detect_communities,
    filter_reciprocal_edges,
)
from src.phase4_network.centrality import compute_centrality_features

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Faz 2-3'te öne çıkan örnekler: 3 "gerçek edit war" imzalı tartışmalı sayfa +
# kontrastı görmek için 1 kontrol (vandalizm-çürümesi) sayfası
VISUALIZE_PAGES = {"Abortion", "Taiwan", "Kurdistan", "Igneous rock"}


def load_all_revisions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM revisions_clean", conn)


def visualize_graph(graph, communities, title, output_path):
    if graph.number_of_edges() == 0:
        return

    pos = nx.spring_layout(graph, seed=42, k=0.6)
    community_ids = sorted(set(communities.values()))
    cmap = plt.get_cmap("tab10")
    # topluluğu olmayan (tek seferlik/karşılıklı olmayan) düğümler gri:
    # gerçek "çekişme çekirdeği"nin dışında kalan, çevresel katılımcılar
    GRAY = (0.75, 0.75, 0.75, 1.0)
    colors = [
        cmap(community_ids.index(communities[n]) % 10) if n in communities else GRAY
        for n in graph.nodes()
    ]
    degree = nx.degree_centrality(graph)
    sizes = [300 + 4000 * degree[n] for n in graph.nodes()]

    top_nodes = sorted(degree.items(), key=lambda x: -x[1])[:8]
    labels = {n: n for n, _ in top_nodes}

    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(graph, pos, alpha=0.25, arrows=True, arrowsize=8, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title(f"{title} — editör revert ağı ({len(community_ids)} topluluk)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_revisions()

    editor_rows = []
    page_rows = []

    for page_title, group in df.groupby("page_title"):
        page_label = group["page_label"].iloc[0]
        graph = build_revert_graph(group)

        if graph.number_of_nodes() == 0:
            page_rows.append(
                {
                    "page_title": page_title,
                    "page_label": page_label,
                    "num_editors_in_graph": 0,
                    "num_edges": 0,
                    "num_editors_in_core_conflict": 0,
                    "num_communities": 0,
                    "largest_community_share": 0.0,
                }
            )
            continue

        # Topluluk tespitini TÜM grafta değil, sadece karşılıklı (gerçek
        # çekişme) kenarların olduğu alt-grafta yapıyoruz — tek seferlik
        # vandalizm kenarları "sahte topluluklar" yaratmasın diye.
        reciprocal_graph = filter_reciprocal_edges(graph)
        communities = detect_communities(reciprocal_graph) if reciprocal_graph.number_of_edges() else {}

        centrality_df = compute_centrality_features(graph)
        centrality_df["page_title"] = page_title
        centrality_df["page_label"] = page_label
        centrality_df["community_id"] = centrality_df["user"].map(communities).fillna(-1).astype(int)
        editor_rows.append(centrality_df)

        community_sizes = pd.Series(communities).value_counts()
        largest_share = (
            community_sizes.max() / community_sizes.sum() if len(community_sizes) else 0.0
        )

        page_rows.append(
            {
                "page_title": page_title,
                "page_label": page_label,
                "num_editors_in_graph": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "num_editors_in_core_conflict": reciprocal_graph.number_of_nodes(),
                "num_communities": int(community_sizes.shape[0]),
                "largest_community_share": largest_share,
            }
        )

        if page_title in VISUALIZE_PAGES:
            safe_name = page_title.lower().replace(" ", "_")
            visualize_graph(
                graph, communities, page_title, FIGURES_DIR / f"network_{safe_name}.png"
            )
            print(f"  görselleştirildi: {page_title}")

    editor_features = pd.concat(editor_rows, ignore_index=True) if editor_rows else pd.DataFrame()
    page_features = pd.DataFrame(page_rows)
    page_features["core_conflict_ratio"] = (
        page_features["num_editors_in_core_conflict"] / page_features["num_editors_in_graph"]
    ).fillna(0.0)

    with sqlite3.connect(DB_PATH) as conn:
        editor_features.to_sql("editor_network_features", conn, if_exists="replace", index=False)
        page_features.to_sql("page_network_features", conn, if_exists="replace", index=False)

    print(f"\neditor_network_features: {len(editor_features)} satır")
    print(f"page_network_features: {len(page_features)} satır")

    print("\n--- Grup karşılaştırması: ortalama 'çekişme çekirdeği' oranı ve topluluk sayısı ---")
    print(
        page_features.groupby("page_label")[
            ["core_conflict_ratio", "num_editors_in_core_conflict", "num_communities"]
        ].mean()
    )

    print("\n--- En büyük 'çekişme çekirdeği' oranına sahip 5 sayfa ---")
    top = page_features.sort_values("core_conflict_ratio", ascending=False).head(5)
    print(
        top[
            ["page_title", "page_label", "core_conflict_ratio", "num_editors_in_core_conflict", "num_editors_in_graph"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
