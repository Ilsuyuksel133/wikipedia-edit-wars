# Wikipedia Edit Wars & Tartışma Ağları Analizi

Wikipedia sayfalarındaki düzenleme çatışmalarını (edit wars) network/graph analizi,
zaman serisi ve NLP ile inceleyip iki tahmin modeli kurmak:

1. **Editör seviyesi**: Bir düzenlemenin geri alınıp alınmayacağını tahmin etmek
2. **Sayfa seviyesi**: Bir sayfanın yakında edit war'a gireceğini tahmin etmek

## Proje Yapısı

```
wikipedia-edit-wars/
├── config/                    # Sayfa listeleri, API ayarları
├── data/
│   ├── raw/                   # SQLite DB, ham API çıktıları
│   ├── processed/             # Temizlenmiş / feature'lı tablolar
│   └── external/              # Dış olay verileri (seçimler, krizler vb.)
├── src/
│   ├── db/                    # SQLite şema + bağlantı yardımcıları
│   ├── phase1_collection/     # Faz 1: Wikipedia API veri toplama
│   ├── phase2_cleaning/       # Faz 2: Temizlik, revert tespiti, feature çıkarımı
│   ├── phase3_timeseries/     # Faz 3: Zaman serisi analizi
│   ├── phase4_network/        # Faz 4: Network / graph analizi
│   ├── phase5_nlp/            # Faz 5: NLP katmanı (sentiment, TF-IDF)
│   ├── phase6_model_revert/   # Faz 6: Model 1 — Revert prediction
│   └── phase7_model_editwar/  # Faz 7: Model 2 — Edit war öngörüsü
├── notebooks/                 # Keşif ve sunum notebook'ları
├── models/                    # Kaydedilmiş model dosyaları
├── reports/figures/           # Sentez raporu ve görselleştirmeler (Faz 8)
└── tests/
```

## Fazlar

| Faz | İçerik | Durum |
|---|---|---|
| 1 | Veri Toplama | ⬜ |
| 2 | Veri Temizleme & Özellik Çıkarımı | ⬜ |
| 3 | Zaman Serisi Analizi | ⬜ |
| 4 | Network Analizi | ⬜ |
| 5 | NLP Katmanı | ⬜ |
| 6 | Model 1 — Revert Prediction | ⬜ |
| 7 | Model 2 — Edit War Öngörüsü | ⬜ |
| 8 | Sentez & Sunum | ⬜ |

## Kurulum

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Veritabanı

`data/raw/wikipedia_edit_wars.db` (SQLite) — şema `src/db/schema.py` içinde tanımlı.

Ana tablo: `revisions` (page_id, user, timestamp, size_diff, comment, is_revert, ...)
