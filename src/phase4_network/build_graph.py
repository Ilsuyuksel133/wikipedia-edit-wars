"""Editör etkileşim grafı: düğümler = editörler, kenarlar = 'A, B'nin
editini geri aldı'.
"""

import networkx as nx


def build_revert_graph(revisions_df):
    """
    revisions_df: en az şu sütunları içermeli: rev_id, user, is_reverted,
    reverted_by_rev_id (Faz 2'nin ürettiği revisions_clean formatı, TEK bir
    sayfaya ait olmalı — revert ilişkileri sayfa içinde anlamlıdır).

    Kenar A -> B: A, B'nin editini geri almış. Ağırlık = kaç kez.
    Aynı kişinin kendi editini "geri alması" (örn. kendi kendini düzeltmesi)
    anlamsız olduğu için atlanır.
    """
    rev_id_to_user = dict(zip(revisions_df["rev_id"], revisions_df["user"]))

    reverted = revisions_df[revisions_df["is_reverted"] == 1].copy()
    reverted["reverter_user"] = reverted["reverted_by_rev_id"].map(rev_id_to_user)
    reverted = reverted[reverted["reverter_user"] != reverted["user"]]

    edge_weights = reverted.groupby(["reverter_user", "user"]).size()

    graph = nx.DiGraph()
    for (reverter, reverted_user), weight in edge_weights.items():
        graph.add_edge(reverter, reverted_user, weight=int(weight))

    return graph


def filter_reciprocal_edges(graph):
    """Sadece KARŞILIKLI kenarları (A->B VE B->A ikisi de var) içeren bir
    alt-graf döndürür.

    Neden: tek yönlü, tek seferlik kenarlar genelde "bir vandalı geri aldım"
    demek — gerçek bir taraflaşma göstermez. Karşılıklı kenarlar ise iki
    editörün birbirini tekrar tekrar geri aldığı, gerçek bir çekişmenin
    izi. Topluluk tespitini bu alt-grafta yapmak, "kaç farklı tek seferlik
    vandal var" gürültüsünü eleyip asıl "taraflara" odaklanmamızı sağlar.
    """
    reciprocal = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        if graph.has_edge(v, u):
            reciprocal.add_edge(u, v, weight=data.get("weight", 1))
    return reciprocal


def detect_communities(graph):
    """Louvain algoritmasıyla topluluk (taraf) tespiti.

    Louvain yönsüz graf üzerinde çalışır; A->B ve B->A kenarlarının
    ağırlıklarını toplayarak yönsüz bir graf oluşturuyoruz (iki editör
    birbirini ne kadar çok geri almışsa aralarındaki "bağ" o kadar güçlü).
    """
    import community as community_louvain

    undirected = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1)
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += w
        else:
            undirected.add_edge(u, v, weight=w)

    if undirected.number_of_edges() == 0:
        return {node: 0 for node in graph.nodes()}

    return community_louvain.best_partition(undirected, weight="weight", random_state=42)
