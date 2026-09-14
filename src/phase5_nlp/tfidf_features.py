"""TF-IDF ile en sık geçen kelime/ifadeler ve 'çatışmalı düzenleme' sinyali."""

from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(comments, max_features=500):
    """Edit summary'ler üzerinde TF-IDF vektörleyici eğitir.

    TODO: controversial vs control sayfa gruplarına göre ayrı ayrı en yüksek
    ağırlıklı terimler karşılaştırılacak.
    """
    raise NotImplementedError
