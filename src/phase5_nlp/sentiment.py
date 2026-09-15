"""Edit summary metinlerinden ton/duygu analizi ve çatışma anahtar kelime sinyali."""

import re

import nltk

_analyzer = None

# Wikipedia edit summary'lerinde çatışma/anlaşmazlık belirten yaygın ifadeler.
CONFLICT_KEYWORDS = [
    "revert", "rv", "undo", "undid", "vandal", "vandalism", "pov", "npov",
    "please stop", "discuss", "dispute", "edit war", "unconstructive",
    "not neutral", "biased", "original research", "unsourced", "citation needed",
    "warning", "restore", "rvv",
]
_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONFLICT_KEYWORDS) + r")\b", re.IGNORECASE
)

_WIKILINK_PATTERN = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]|\[\[([^\]]*)\]\]")

# Rollback/Twinkle/undo araçlarının otomatik ürettiği, tüm sayfalarda aynı
# şekilde görünen kalıp metinler — gerçek insan yorumu değil, "imza".
_BOILERPLATE_PATTERNS = [
    re.compile(
        r"Reverted \d*\s*(pending\s+)?edits? by .*?"
        r"(identified as vandalism )?to last (revision|version) by .*?\.?",
        re.IGNORECASE,
    ),
    re.compile(r"Undid revision \d+ by .*?\(talk\)", re.IGNORECASE),
    # Twinkle/Huggle/ClueBot/AWB gibi araçların bıraktığı kısa "imza" etiketleri
    re.compile(r"\busing\s+(TW|HG|STiki|RedWarn|AWB|CB|Twinkle|Huggle)\b\.?", re.IGNORECASE),
    re.compile(r"\((TW|HG|STiki|RedWarn|AWB|CB)\)", re.IGNORECASE),
]


def strip_tool_boilerplate(comment_text):
    """Wikilink işaretlemesini ve bilinen araç şablonlarını temizler; geriye
    (varsa) editörün kendi yazdığı serbest metni bırakır."""
    if not isinstance(comment_text, str):
        return ""

    def _unwrap_link(m):
        return m.group(1) or m.group(2) or ""

    text = _WIKILINK_PATTERN.sub(_unwrap_link, comment_text)
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_sentiment(comment_text):
    """Edit summary metnine VADER ile 'compound' duygu skoru atar: -1 (çok
    negatif/öfkeli) ile +1 (çok pozitif) arasında. Boş/NaN yorum -> 0.0 (nötr)."""
    if not isinstance(comment_text, str) or not comment_text.strip():
        return 0.0
    return _get_analyzer().polarity_scores(comment_text)["compound"]


def count_conflict_keywords(comment_text):
    """Edit summary'de kaç farklı çatışma anahtar kelimesi geçtiğini sayar."""
    if not isinstance(comment_text, str) or not comment_text.strip():
        return 0
    return len(_KEYWORD_PATTERN.findall(comment_text))
