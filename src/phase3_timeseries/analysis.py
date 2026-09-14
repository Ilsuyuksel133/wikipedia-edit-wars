"""Sayfa başına düzenleme yoğunluğunun zaman içindeki değişimi."""

import matplotlib

matplotlib.use("Agg")  # ekrana değil dosyaya çizdirmek için (arka planda çalışırken gerekli)
import matplotlib.pyplot as plt
import pandas as pd


def edit_intensity_over_time(revisions_df, freq="D"):
    """
    revisions_df: en az 'timestamp' sütunu içermeli (tek sayfaya ait).
    freq: pandas frekans kodu — 'D' (gün), 'W' (hafta).

    Döndürür: tam tarih aralığını kapsayan bir Series (edit olmayan
    periyotlar 0 olarak görünür, grafikte "boşluk" değil "sıfır" olsun diye).
    """
    ts = pd.to_datetime(revisions_df["timestamp"]).sort_values()
    counts = pd.Series(1, index=ts)
    return counts.resample(freq).sum()


def plot_intensity_timeline(revisions_df, title, output_path, freq="W"):
    """Zaman çizgisi grafiği: belirtilen frekansta edit sayısı."""
    series = edit_intensity_over_time(revisions_df, freq=freq)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index, series.values, linewidth=1, color="#c0392b")
    ax.fill_between(series.index, series.values, alpha=0.2, color="#c0392b")
    ax.set_title(f"{title} — {'haftalık' if freq == 'W' else 'günlük'} edit sayısı")
    ax.set_ylabel("edit sayısı")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_calendar_heatmap(revisions_df, title, output_path):
    """Bir sayfa için GitHub-tarzı takvim heatmap'i üretir: satırlar haftanın
    günleri, sütunlar haftalar; renk o günkü edit sayısını gösterir."""
    ts = pd.to_datetime(revisions_df["timestamp"])
    daily = ts.dt.floor("D").value_counts().sort_index()

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)

    cal = pd.DataFrame({"date": daily.index, "count": daily.values})
    iso = cal["date"].dt.isocalendar()
    cal["yw"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    cal["weekday"] = iso["day"]  # 1=Pzt ... 7=Paz

    pivot = cal.pivot_table(index="weekday", columns="yw", values="count", fill_value=0)
    pivot = pivot[sorted(pivot.columns, key=lambda c: tuple(map(int, c.split("-"))))]

    fig_width = max(8, pivot.shape[1] * 0.15)
    fig, ax = plt.subplots(figsize=(fig_width, 2.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"])
    ax.set_xticks([])
    ax.set_title(f"{title} — günlük edit yoğunluğu (takvim görünümü)")
    fig.colorbar(im, ax=ax, label="edit sayısı", fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
