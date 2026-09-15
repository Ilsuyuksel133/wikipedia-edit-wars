"""
Faz 6 ana script: Model 1 (Revert Prediction) feature matrisini kurar, üç
modeli eğitir/karşılaştırır, feature importance ve SHAP özetini üretir.

Kullanım:
    python -m src.phase6_model_revert.run_phase6
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import PROJECT_ROOT
from src.phase6_model_revert.train import build_feature_matrix, train_and_evaluate

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    print("1/3 Feature matrisi kuruluyor...")
    X, y, meta = build_feature_matrix()
    print(f"     {len(X)} revizyon, {X.shape[1]} feature")
    print(f"     Sınıf dengesi: is_reverted=1 oranı = {y.mean():.3f}")

    print("\n2/3 Modeller eğitiliyor (Logistic Regression -> Random Forest -> XGBoost)...")
    results, feature_importance, splits = train_and_evaluate(X, y)
    _, X_test, _, y_test = splits

    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(f"ROC-AUC: {res['roc_auc']:.4f}")
        print("Confusion matrix [[TN FP] [FN TP]]:")
        print(res["confusion_matrix"])
        print(res["classification_report"])

    print("\n--- Random Forest feature importance ---")
    print(feature_importance.to_string())

    print("\n3/3 SHAP özeti hesaplanıyor (Random Forest, 500 örneklik alt küme)...")
    try:
        import shap

        rf = results["random_forest"]["model"]
        sample = X_test.sample(min(500, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(sv, sample, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary_model1.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("     kaydedildi: reports/figures/shap_summary_model1.png")
    except Exception as exc:
        print(f"     SHAP hesaplanamadı: {exc}")


if __name__ == "__main__":
    main()
