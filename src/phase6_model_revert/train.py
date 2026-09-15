"""
Model 1 — Revert Prediction (Editör Seviyesi)

Hedef: bir düzenleme yapıldığı anda geri alınıp alınmayacağını tahmin etmek.

ÖNEMLİ (veri sızıntısı): Faz 2/4'teki page_conflict_scores ve
editor_network_features tabloları sayfanın TÜM geçmişinden (gelecekteki
revizyonlar dahil) hesaplandı — bu yüzden burada KULLANILMIYOR. Onun
yerine sayfa aktivitesi, her revizyondan ÖNCEKİ 7 günlük kayan pencereyle
kendimiz hesaplanıyor (bkz. _add_rolling_page_activity).

Feature'lar: editör geçmişi (Faz 2, zaten sızıntısız), düzenleme boyutu,
edit summary/NLP sinyalleri (Faz 5, revizyon anında bilinir), kayan
pencereli sayfa aktivitesi, saat.

Modeller: Logistic Regression (baseline) -> Random Forest -> XGBoost
Değerlendirme: ROC-AUC, confusion matrix, precision/recall, feature
importance + SHAP.
"""

import sqlite3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import DB_PATH

FEATURE_COLS = [
    "size_diff",
    "is_revert",
    "anon",
    "editor_prior_edit_count",
    "editor_prior_revert_action_rate",
    "editor_prior_reverted_rate",
    "has_comment",
    "comment_length",
    "sentiment_score",
    "conflict_keyword_count",
    "is_auto_generated_summary",
    "page_edits_last_7d",
    "page_revert_rate_last_7d",
    "hour_of_day",
]


def _load_base():
    with sqlite3.connect(DB_PATH) as conn:
        revisions = pd.read_sql_query("SELECT * FROM revisions_clean", conn)
        nlp = pd.read_sql_query("SELECT * FROM revisions_nlp", conn)
    return revisions.merge(nlp, on="rev_id", how="left")


def _add_rolling_page_activity(df, window="7D"):
    """Her revizyondan ÖNCEKİ `window` süresinde sayfada kaç edit/revert
    olmuş — closed='left' ile şu anki revizyonun kendisi pencereye dahil
    edilmiyor (gelecek sızıntısı olmasın diye)."""
    df = df.sort_values(["page_id", "timestamp"]).reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["timestamp"])

    edits_parts, revert_rate_parts = [], []
    for _, group in df.groupby("page_id", sort=False):
        g = group.set_index("ts")
        edits_roll = g["rev_id"].rolling(window, closed="left").count()
        reverts_roll = g["is_revert"].rolling(window, closed="left").sum()
        edits_parts.append(edits_roll)
        revert_rate_parts.append((reverts_roll / edits_roll).fillna(0.0))

    df["page_edits_last_7d"] = pd.concat(edits_parts).fillna(0).values
    df["page_revert_rate_last_7d"] = pd.concat(revert_rate_parts).fillna(0).values
    return df


def build_feature_matrix():
    """Faz 2, 5 çıktılarını ve kayan pencereli sayfa aktivitesini birleştirip
    model girdisi oluşturur.

    Döndürür: (X, y, meta) — X: feature DataFrame, y: is_reverted hedefi,
    meta: rev_id/page/user/timestamp (analiz için, modele verilmiyor).
    """
    df = _load_base()
    df = _add_rolling_page_activity(df, window="7D")
    df["hour_of_day"] = df["ts"].dt.hour

    X = df[FEATURE_COLS].fillna(0)
    y = df["is_reverted"].astype(int)
    meta = df[["rev_id", "page_id", "page_title", "page_label", "timestamp", "user"]]
    return X, y, meta


def _evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "model": model,
    }


def train_and_evaluate(X, y):
    """Üç modeli sırayla eğitir ve karşılaştırır.

    Döndürür: (results_dict, feature_importance, (X_train, X_test, y_train, y_test))
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    results["logistic_regression"] = _evaluate(log_reg, X_test_scaled, y_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = _evaluate(rf, X_test, y_test)

    try:
        from xgboost import XGBClassifier

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        results["xgboost"] = _evaluate(xgb, X_test, y_test)
    except Exception as exc:
        print(f"  ! XGBoost atlandı: {exc}")

    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    return results, feature_importance, (X_train, X_test, y_train, y_test)
