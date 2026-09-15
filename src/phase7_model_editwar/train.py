"""
Model 2 — Edit War Öngörüsü (Sayfa Seviyesi)

Hedef: bir sayfanın önümüzdeki 7 gün içinde edit war'a girip girmeyeceğini
tahmin etmek.

Panel yapısı: Faz 6'daki "her revizyon bir satır"ın aksine, burada
"her (sayfa, hafta) bir satır". Hedef = GELECEK haftanın edit-war durumu
(Faz 4'teki doğrulanmış tanım: o hafta içinde >=1 karşılıklı revert çifti).

Not: Faz 4'ün network/çatışma özetlerini burada kullanmak Faz 6'daki gibi
sızıntı DEĞİL — çünkü "şu ana kadarki durum" ile "gelecek hafta"yı tahmin
ediyoruz; kümülatif özellikler her hafta sonu itibarıyla hesaplanıyor.

Modeller: Logistic Regression -> Random Forest -> XGBoost
Değerlendirme: Model 1 ile aynı metrikler + KRONOLOJİK train/test split.
"""

import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config.settings import DB_PATH

FEATURE_COLS = [
    "edits",
    "reverts",
    "unique_editors",
    "revert_rate",
    "reciprocal_pairs",
    "sentiment_avg",
    "conflict_kw_avg",
    "edits_trend_3w",
    "revert_rate_trend_3w",
    "cumulative_reciprocal_pairs",
    "is_controversial",
]


def _load_base():
    with sqlite3.connect(DB_PATH) as conn:
        revisions = pd.read_sql_query("SELECT * FROM revisions_clean", conn)
        nlp = pd.read_sql_query(
            "SELECT rev_id, sentiment_score, conflict_keyword_count FROM revisions_nlp", conn
        )
    df = revisions.merge(nlp, on="rev_id", how="left")
    df["ts"] = pd.to_datetime(df["timestamp"])
    return df


def _weekly_reciprocal_pairs(week_df):
    """Bu hafta İÇİNDE gerçekleşen revert eylemlerinden (reverter, geri
    alınan editör) çiftlerini kurar, karşılıklı olanların sayısını döner."""
    reverts = week_df[week_df["is_revert"] == 1]
    if reverts.empty:
        return 0

    rev_id_to_user = dict(zip(week_df["rev_id"], week_df["user"]))
    reverted = week_df[week_df["is_reverted"] == 1].copy()
    reverted = reverted[reverted["reverted_by_rev_id"].isin(reverts["rev_id"])]
    reverted["reverter_user"] = reverted["reverted_by_rev_id"].map(rev_id_to_user)

    pairs = {
        (r, u)
        for r, u in zip(reverted["reverter_user"], reverted["user"])
        if r != u and pd.notna(r)
    }
    reciprocal = {frozenset((a, b)) for (a, b) in pairs if (b, a) in pairs}
    return len(reciprocal)


def _build_page_week_panel(page_df):
    page_df = page_df.sort_values("ts")
    start = page_df["ts"].min().floor("D")
    end = page_df["ts"].max().ceil("D")
    weeks = pd.date_range(start, end, freq="7D")

    records = []
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=7)
        week_df = page_df[(page_df["ts"] >= week_start) & (page_df["ts"] < week_end)]

        edits = len(week_df)
        reverts = int(week_df["is_revert"].sum())
        reciprocal = _weekly_reciprocal_pairs(week_df)

        records.append(
            {
                "week_start": week_start,
                "edits": edits,
                "reverts": reverts,
                "unique_editors": week_df["user"].nunique(),
                "revert_rate": reverts / edits if edits else 0.0,
                "reciprocal_pairs": reciprocal,
                "has_conflict_week": int(reciprocal >= 1),
                "sentiment_avg": week_df["sentiment_score"].mean() if edits else 0.0,
                "conflict_kw_avg": week_df["conflict_keyword_count"].mean() if edits else 0.0,
            }
        )

    wdf = pd.DataFrame(records)
    wdf["edits_trend_3w"] = wdf["edits"].rolling(3, min_periods=1).mean()
    wdf["revert_rate_trend_3w"] = wdf["revert_rate"].rolling(3, min_periods=1).mean()
    wdf["cumulative_reciprocal_pairs"] = wdf["reciprocal_pairs"].cumsum()

    # HEDEF: önümüzdeki 4 hafta içinde (bu hafta HARİÇ) en az bir çatışma
    # haftası var mı. Tek haftalık ufuk denendi ama pozitif örnek o kadar
    # seyrekti (~180 hafta / 40 sayfa x yıllar) ki modeller güvenilir
    # öğrenemedi — 4 haftaya genişletmek hem daha az gürültülü bir hedef,
    # hem de pratikte hâlâ kullanışlı bir "erken uyarı" ufku.
    conflict = wdf["has_conflict_week"].to_numpy()
    n = len(conflict)
    horizon = 4
    labels = np.full(n, np.nan)
    for i in range(n - horizon):
        labels[i] = 1.0 if conflict[i + 1 : i + 1 + horizon].max() >= 1 else 0.0
    wdf["edit_war_next_4w"] = labels
    return wdf


def build_feature_matrix():
    """Her (sayfa, hafta) satırı için özellik + hedef üretir.

    Döndürür: (X, y, meta) — meta: page_id/title/label/week_start
    (kronolojik split ve analiz için, modele verilmiyor).
    """
    df = _load_base()

    panels = []
    for (page_id, page_title, page_label), page_df in df.groupby(
        ["page_id", "page_title", "page_label"]
    ):
        wdf = _build_page_week_panel(page_df)
        wdf["page_id"] = page_id
        wdf["page_title"] = page_title
        wdf["page_label"] = page_label
        panels.append(wdf)

    full = pd.concat(panels, ignore_index=True)
    full = full.dropna(subset=["edit_war_next_4w"])
    # Bazı sayfaların (örn. az düzenlenen "Basalt") son 2000 revizyonu
    # onlarca yıla yayılıyor -> çoğu hafta hiç edit yok. Bu "ölü haftaları"
    # (rolling/cumulative trend'leri hesapladıktan SONRA) eliyoruz: "hiç
    # düzenlenmeyen bir sayfa gelecek hafta savaşa girer mi" sorusu triviyal
    # ve veri setini anlamsızca dengesizleştiriyor.
    full = full[full["edits"] > 0]
    full["is_controversial"] = (full["page_label"] == "controversial").astype(int)

    X = full[FEATURE_COLS].fillna(0)
    y = full["edit_war_next_4w"].astype(int)
    meta = full[["page_id", "page_title", "page_label", "week_start"]]
    return X, y, meta


def chronological_train_test_split(X, y, timestamps, test_size=0.2):
    """Gelecek veri sızıntısını önlemek için ZAMAN bazlı split: tüm
    (sayfa, hafta) satırlarını takvim zamanına göre sıralayıp son
    test_size oranını test seti yapar (rastgele karıştırma yok)."""
    order = np.argsort(timestamps.values)
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)

    split_idx = int(len(X_sorted) * (1 - test_size))
    return (
        X_sorted.iloc[:split_idx],
        X_sorted.iloc[split_idx:],
        y_sorted.iloc[:split_idx],
        y_sorted.iloc[split_idx:],
    )


def _evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "model": model,
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Üç modeli sırayla eğitir ve karşılaştırır (kronolojik split zaten
    yapılmış olarak verilir)."""
    results = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    results["logistic_regression"] = _evaluate(log_reg, X_test_scaled, y_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = _evaluate(rf, X_test, y_test)

    try:
        from xgboost import XGBClassifier

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        results["xgboost"] = _evaluate(xgb, X_test, y_test)
    except Exception as exc:
        print(f"  ! XGBoost atlandı: {exc}")

    feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(
        ascending=False
    )

    return results, feature_importance
