"""Editör güvenilirlik metrikleri ve sayfa başına çatışma skoru.

Önemli ilke: her revizyon için hesaplanan feature'lar SADECE o revizyondan
ÖNCEKİ bilgiyi kullanmalı (gelecek veri sızıntısı olmasın diye) — bu yüzden
revizyonları zaman sırasına göre tek tek gezip sayaçları o sırada güncelliyoruz.
"""

import pandas as pd


def compute_editor_features(revisions_df):
    """
    revisions_df: en az şu sütunları içermeli: user, timestamp, is_revert,
    is_reverted.

    Ekler:
    - editor_prior_edit_count: bu editörün o ana kadarki toplam edit sayısı
    - editor_prior_revert_action_rate: geçmiş editlerinin ne kadarı
      başkasını geri alma eylemiydi
    - editor_prior_reverted_rate: geçmiş editlerinin ne kadarı sonradan
      başkası tarafından geri alındı (güvenilirlik göstergesi)
    """
    df = revisions_df.sort_values("timestamp").reset_index(drop=True)

    edit_count = {}
    revert_action_count = {}
    reverted_count = {}

    prior_edits, prior_revert_action_rate, prior_reverted_rate = [], [], []

    for _, row in df.iterrows():
        user = row["user"]
        n = edit_count.get(user, 0)

        prior_edits.append(n)
        prior_revert_action_rate.append(revert_action_count.get(user, 0) / n if n else 0.0)
        prior_reverted_rate.append(reverted_count.get(user, 0) / n if n else 0.0)

        edit_count[user] = n + 1
        if row.get("is_revert"):
            revert_action_count[user] = revert_action_count.get(user, 0) + 1
        if row.get("is_reverted"):
            reverted_count[user] = reverted_count.get(user, 0) + 1

    df["editor_prior_edit_count"] = prior_edits
    df["editor_prior_revert_action_rate"] = prior_revert_action_rate
    df["editor_prior_reverted_rate"] = prior_reverted_rate
    return df


def compute_page_conflict_score(page_revisions_df):
    """
    page_revisions_df: TEK bir sayfaya ait revizyonlar. En az şu sütunları
    içermeli: rev_id, user, timestamp, anon, is_revert, is_reverted,
    reverted_by_rev_id.

    Döndürür: dict —
    - revert_rate: revizyonların ne kadarı revert eylemi
    - unique_editors: kaç farklı kişi düzenlemiş
    - edit_burst_score: en yoğun günde kaç edit yapılmış ("ateşlenme" göstergesi)
    - revert_concentration_hhi: revert eylemleri az sayıda kişide mi
      yoğunlaşmış (1'e yakın) yoksa çok kişiye mi dağılmış (0'a yakın)?
      (Herfindahl-Hirschman Index: her kişinin revert payının karelerinin toplamı)
    - anon_reverted_ratio: geri alınan editlerin ne kadarı anonim
      kullanıcılardan geldi (yüksekse vandalizm belirtisi)
    - reciprocal_revert_pairs: A'nın B'yi, B'nin de A'yı geri aldığı
      kaç FARKLI editör çifti var — gerçek edit war'ın en net imzası
    """
    total = len(page_revisions_df)
    revert_rate = page_revisions_df["is_revert"].sum() / total if total else 0.0
    unique_editors = page_revisions_df["user"].nunique()

    ts = pd.to_datetime(page_revisions_df["timestamp"])
    daily_counts = ts.groupby(ts.dt.floor("D")).size()
    edit_burst_score = int(daily_counts.max()) if len(daily_counts) else 0

    reverters = page_revisions_df.loc[page_revisions_df["is_revert"] == 1, "user"]
    if len(reverters):
        shares = reverters.value_counts(normalize=True)
        revert_concentration_hhi = float((shares**2).sum())
    else:
        revert_concentration_hhi = 0.0

    reverted = page_revisions_df[page_revisions_df["is_reverted"] == 1].copy()
    anon_reverted_ratio = float(reverted["anon"].mean()) if len(reverted) else 0.0

    rev_id_to_user = dict(zip(page_revisions_df["rev_id"], page_revisions_df["user"]))
    reverted["reverter_user"] = reverted["reverted_by_rev_id"].map(rev_id_to_user)
    pair_counts = reverted.groupby(["reverter_user", "user"]).size()
    pairs = set(pair_counts.index)
    reciprocal_pairs = {
        frozenset((a, b)) for (a, b) in pairs if a != b and (b, a) in pairs
    }

    return {
        "revert_rate": revert_rate,
        "unique_editors": unique_editors,
        "edit_burst_score": edit_burst_score,
        "revert_concentration_hhi": revert_concentration_hhi,
        "anon_reverted_ratio": anon_reverted_ratio,
        "reciprocal_revert_pairs": len(reciprocal_pairs),
    }
