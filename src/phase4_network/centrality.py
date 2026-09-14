"""Merkezilik metrikleri: betweenness, degree centrality."""

import networkx as nx
import pandas as pd


def compute_centrality_features(graph):
    """Her düğüm (editör) için merkezilik metrikleri hesaplar.

    Döndürür: DataFrame (user, degree_centrality, in_degree_centrality,
    out_degree_centrality, betweenness_centrality)

    - in_degree: kaç farklı kişi tarafından geri alınmış ("hedef" olma)
    - out_degree: kaç farklı kişiyi geri almış ("saldıran" olma)
    - betweenness: farklı çatışma "kümeleri" arasında köprü kurma derecesi

    Not: betweenness_centrality'ye kenar ağırlığını doğrudan vermiyoruz.
    networkx weight parametresini "mesafe" olarak yorumluyor — yani yüksek
    ağırlık (çok revert) "daha uzak/zayıf bağ" anlamına gelirdi, tam tersi
    olması gerekirken. Bu yüzden burada ağırlıksız (sadece bağlantı var mı
    yok mu) versiyonu kullanıyoruz.
    """
    degree = nx.degree_centrality(graph)
    in_degree = nx.in_degree_centrality(graph)
    out_degree = nx.out_degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)

    rows = [
        {
            "user": node,
            "degree_centrality": degree[node],
            "in_degree_centrality": in_degree[node],
            "out_degree_centrality": out_degree[node],
            "betweenness_centrality": betweenness[node],
        }
        for node in graph.nodes()
    ]
    return pd.DataFrame(rows)
