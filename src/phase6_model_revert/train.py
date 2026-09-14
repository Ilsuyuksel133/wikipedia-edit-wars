"""
Model 1 — Revert Prediction (Editör Seviyesi)

Hedef: bir düzenleme yapıldığı anda geri alınıp alınmayacağını tahmin etmek.

Feature'lar: editör geçmişi, düzenleme boyutu, edit summary varlığı/uzunluğu/
sentiment, sayfanın çatışma skoru, network centrality, zamanlama.

Modeller: Logistic Regression (baseline) -> Random Forest -> XGBoost/LightGBM
Değerlendirme: Precision/Recall, ROC-AUC, confusion matrix, SHAP.
"""


def build_feature_matrix():
    """Faz 2, 4, 5 çıktılarını birleştirip model girdisi oluşturur.

    TODO
    """
    raise NotImplementedError


def train_and_evaluate(X, y):
    """Üç modeli sırayla eğitir ve karşılaştırır.

    TODO
    """
    raise NotImplementedError
