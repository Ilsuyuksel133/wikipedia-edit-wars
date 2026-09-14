"""
Model 2 — Edit War Öngörüsü (Sayfa Seviyesi)

Hedef: bir sayfanın önümüzdeki 7 gün içinde edit war'a girip girmeyeceğini
tahmin etmek.

Feature'lar: son N günlük edit yoğunluğu trendi, network yapısı (editör
çeşitliliği, topluluk polarizasyonu), NLP sentiment trendi, sayfa kategorisi.

Değerlendirme: Model 1 ile aynı metrik seti + kronolojik train/test split
(veri sızıntısı olmaması için).
"""


def build_feature_matrix():
    """Faz 3, 4, 5 çıktılarını sayfa-zaman penceresi bazında birleştirir.

    TODO
    """
    raise NotImplementedError


def chronological_train_test_split(X, y, timestamps, test_size=0.2):
    """Gelecek veri sızıntısını önlemek için zaman bazlı split.

    TODO
    """
    raise NotImplementedError


def train_and_evaluate(X, y):
    """Üç modeli sırayla eğitir ve karşılaştırır.

    TODO
    """
    raise NotImplementedError
