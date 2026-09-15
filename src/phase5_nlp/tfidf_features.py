"""TF-IDF ile en sık geçen kelime/ifadeler ve grup bazlı dil karşılaştırması."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(comments, max_features=500):
    """Edit summary'ler üzerinde TF-IDF vektörleyici eğitir.

    comments: metin listesi/Series (boş değerler '' olarak geçmeli).
    Döndürür: (vectorizer, tfidf_matrix)
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=3)
    matrix = vectorizer.fit_transform(comments)
    return vectorizer, matrix


def top_terms_by_group(labels, group_value, vectorizer, matrix, top_n=20):
    """Belirli bir gruba (örn. 'controversial') ait satırların ortalama
    TF-IDF ağırlığına göre en yüksek terimlerini döndürür."""
    mask = np.asarray(labels) == group_value
    group_matrix = matrix[mask]
    mean_scores = np.asarray(group_matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = mean_scores.argsort()[::-1][:top_n]
    return [(terms[i], float(mean_scores[i])) for i in order]
