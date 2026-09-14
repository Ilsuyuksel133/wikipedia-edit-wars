"""
Analiz edilecek Wikipedia sayfa listeleri.

Bu listeler başlangıç önerisidir — projeye başlamadan önce gözden geçirip
düzenleyebilirsin. `label` her sayfa için hangi grupta (controversial/control)
olduğunu Faz 1 veri toplama script'ine bildirir.
"""

# Tartışmalı sayfalar: siyaset, tarih, güncel olaylar — yüksek edit war riski beklenen
CONTROVERSIAL_PAGES = [
    "Donald Trump",
    "Israeli–Palestinian conflict",
    "Abortion",
    "Gun control",
    "Brexit",
    "Climate change",
    "COVID-19 vaccine",
    "Vladimir Putin",
    "Armenian Genocide",
    "Kurdistan",
    "Genetically modified organism",
    "Black Lives Matter",
    "Xi Jinping",
    "Russian invasion of Ukraine",
    "Capital punishment",
    "Immigration",
    "Nationalism",
    "Zionism",
    "Taiwan",
    "Critical race theory",
]

# Kontrol (sakin) sayfalar: teknik/bilimsel/tarihsel konular — düşük çatışma beklenen
CONTROL_PAGES = [
    "Photosynthesis",
    "Helium",
    "Sequoia sempervirens",
    "List of prime numbers",
    "Pythagorean theorem",
    "Andromeda Galaxy",
    "Basalt",
    "Honey bee",
    "Water cycle",
    "Periodic table",
    "Mount Everest",
    "Origami",
    "Sourdough",
    "Chess opening",
    "Acoustic guitar",
    "Igneous rock",
    "Great Barrier Reef",
    "Knitting",
    "Weather forecasting",
    "Espresso",
]


def all_pages_with_labels():
    """[(page_title, label), ...] döndürür. label: 'controversial' | 'control'."""
    return [(p, "controversial") for p in CONTROVERSIAL_PAGES] + [
        (p, "control") for p in CONTROL_PAGES
    ]
