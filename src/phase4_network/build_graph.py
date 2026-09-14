"""Editör etkileşim grafı: düğümler = editörler, kenarlar = 'A, B'nin
editini geri aldı'.
"""

import networkx as nx


def build_revert_graph(revisions_df):
    """revert ilişkilerinden yönlü, ağırlıklı bir networkx.DiGraph kurar.

    TODO: revisions_df.reverted_to_rev_id kullanılarak kenarlar eklenecek.
    """
    raise NotImplementedError


def detect_communities(graph):
    """Louvain algoritmasıyla topluluk (taraf) tespiti.

    TODO: python-louvain (community) kütüphanesi kullanılacak.
    """
    raise NotImplementedError
