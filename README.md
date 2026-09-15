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

| Faz | İçerik | Durum | Çalıştırma |
|---|---|---|---|
| 1 | Veri Toplama | ✅ | `python -m src.phase1_collection.fetch_revisions` |
| 2 | Veri Temizleme & Özellik Çıkarımı | ✅ | `python -m src.phase2_cleaning.run_phase2` |
| 3 | Zaman Serisi Analizi | ✅ | `python -m src.phase3_timeseries.run_phase3` |
| 4 | Network Analizi | ✅ | `python -m src.phase4_network.run_phase4` |
| 5 | NLP Katmanı | ✅ | `python -m src.phase5_nlp.run_phase5` |
| 6 | Model 1 — Revert Prediction | ✅ | `python -m src.phase6_model_revert.run_phase6` |
| 7 | Model 2 — Edit War Öngörüsü | ✅ | `python -m src.phase7_model_editwar.run_phase7` |
| 8 | Sentez & Sunum | ✅ | `notebooks/synthesis.ipynb` |

Fazlar birbirine bağımlı, sırayla çalıştırılmalı (her biri bir öncekinin
ürettiği SQLite tablolarını okur).

## Kurulum

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Veritabanı

`data/raw/wikipedia_edit_wars.db` (SQLite, `.gitignore`'da — repoya dahil değil,
her kurulumda Faz 1'den itibaren yeniden üretilir).

| Tablo | Üreten faz | İçerik |
|---|---|---|
| `pages`, `revisions` | 1 | Ham revizyon verisi |
| `revisions_clean` | 2 | Bot filtresi, revert tespiti, editör feature'ları |
| `page_conflict_scores` | 2 | Sayfa başına çatışma metrikleri |
| `editor_network_features`, `page_network_features` | 4 | Merkezilik, topluluk, çekişme çekirdeği |
| `revisions_nlp` | 5 | Sentiment, çatışma anahtar kelimesi |

## Bulgular (Özet)

Detaylı analiz için `notebooks/synthesis.ipynb`. Öne çıkanlar:

**En çatışmalı 5 sayfa** (Faz 4'te doğrulanmış `core_conflict_ratio`'ya göre —
sayfadaki editörlerin ne kadarı gerçek karşılıklı-revert "çekişme çekirdeğinde"):
Taiwan, Abortion, Zionism, Israeli–Palestinian conflict, Russo-Ukrainian war
(2022–present) — hepsi tartışmalı grup, kontrol grubundan hiçbiri ilk 5'te değil.

**Tipik bir edit war'ın anatomisi:** Ham `revert_rate` yanıltıcı çıktı —
kontrol sayfaları (okul ödevi konuları: "List of prime numbers", "Water
cycle") anonim vandalizim yüzünden YÜKSEK revert oranına sahip ama bu
"sığ" (%50+ anonim, dağınık, tek seferlik). Tartışmalı sayfalarda revert
oranı görece düşük ama "derin" (~%13 anonim, az sayıda kayıtlı editör
arasında TEKRARLANAN karşılıklı revert). Gerçek ayırt edici metrik ham
oran değil, **karşılıklı revert çifti sayısı / çekişme çekirdeği oranı**.

**Model karşılaştırması:**

| Model | En iyi | ROC-AUC | Not |
|---|---|---|---|
| 1 — Revert Prediction (revizyon seviyesi) | XGBoost | 0.893 | En güçlü feature: editörün geçmiş "geri alınma" oranı |
| 2 — Edit War Öngörüsü (sayfa-hafta seviyesi) | Logistic Regression | 0.755 | Sinyal gerçek ama precision düşük — 40 sayfalık veri setinde çekişme olayları doğası gereği seyrek |

**Metodolojik dersler** (proje boyunca bulunup düzeltilen sorunlar — hepsi
`git log` içinde detaylı anlatılıyor):
- Wikipedia redirect sayfaları (`redirects=1` olmadan yanlış sayfa çekiliyor)
- Sayfa başına istekler arası bekleme yetmiyor, TÜM istekler arası gerekiyor (429 hatası)
- `rvdir=newer` kalabalık sayfaların sadece en eski (en sakin) dönemini veriyor
- Bot filtresi `bot$` yerine `bot\b` olmalı (örn. "ClueBot NG" kaçıyordu)
- `networkx` betweenness'e ağırlığı doğrudan vermek "mesafe" olarak yorumlanıyor, "güç" değil
- Model 1'de Faz 2/4'ün tüm-geçmiş özet tablolarını kullanmak veri sızıntısı olurdu — kayan pencere gerekti
- Model 2'de 1 haftalık hedef çok seyrekti (%0.8 pozitif) — 4 haftaya genişletmek gerekti
