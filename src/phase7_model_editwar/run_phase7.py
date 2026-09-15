"""
Faz 7 ana script: Model 2 (Edit War Öngörüsü) için sayfa-hafta panelini
kurar, kronolojik split yapar, üç modeli eğitir/karşılaştırır.

Kullanım:
    python -m src.phase7_model_editwar.run_phase7
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import PROJECT_ROOT
from src.phase7_model_editwar.train import (
    build_feature_matrix,
    chronological_train_test_split,
    train_and_evaluate,
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    print("1/3 Sayfa-hafta paneli kuruluyor...")
    X, y, meta = build_feature_matrix()
    print(f"     {len(X)} (sayfa, hafta) satırı, {X.shape[1]} feature")
    print(f"     Sınıf dengesi: edit_war_next_4w=1 oranı = {y.mean():.3f}")
    print(f"     Zaman aralığı: {meta['week_start'].min().date()} -> {meta['week_start'].max().date()}")

    print("\n2/3 Kronolojik train/test split yapılıyor (son %20 test)...")
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        X, y, meta["week_start"], test_size=0.2
    )
    print(f"     train: {len(X_train)} satır, test: {len(X_test)} satır")
    print(f"     train'de pozitif oran: {y_train.mean():.3f}, test'te: {y_test.mean():.3f}")

    print("\nModeller eğitiliyor (Logistic Regression -> Random Forest -> XGBoost)...")
    results, feature_importance = train_and_evaluate(X_train, X_test, y_train, y_test)

    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(f"ROC-AUC: {res['roc_auc']:.4f}")
        print("Confusion matrix [[TN FP] [FN TP]]:")
        print(res["confusion_matrix"])
        print(res["classification_report"])

    print("\n--- Random Forest feature importance ---")
    print(feature_importance.to_string())

    print("\n3/3 SHAP özeti hesaplanıyor (Random Forest)...")
    try:
        import shap

        rf = results["random_forest"]["model"]
        sample = X_test if len(X_test) <= 500 else X_test.sample(500, random_state=42)
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
        plt.savefig(FIGURES_DIR / "shap_summary_model2.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("     kaydedildi: reports/figures/shap_summary_model2.png")
    except Exception as exc:
        print(f"     SHAP hesaplanamadı: {exc}")


if __name__ == "__main__":
    main()
