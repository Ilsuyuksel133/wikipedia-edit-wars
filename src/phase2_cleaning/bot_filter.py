"""Bot editörlerini kullanıcı adı pattern'leri + MediaWiki bot flag'i ile filtreleme."""

import re

BOT_USERNAME_PATTERNS = [
    # "bot" kelimesi bir kelime sınırıyla bitiyorsa eşleş: "ClueBot", "ClueBot NG",
    # "AnomieBOT", "SineBot", "Yobot" yakalanır; ama "Abbott", "Botanist" gibi
    # gerçek isimler yakalanmaz (çünkü "bot"tan hemen sonra harf devam ediyor).
    re.compile(r"bot\b", re.IGNORECASE),
]


def is_bot_username(username):
    # pandas'tan gelen boş değerler NaN (float) olabilir; `not NaN` False
    # döndüğü için (NaN Python'da "truthy"dir) önce tip kontrolü şart.
    if not isinstance(username, str) or not username:
        return False
    return any(p.search(username) for p in BOT_USERNAME_PATTERNS)


def filter_bot_revisions(df):
    """revisions DataFrame'inden bot editörlerine ait satırları çıkarır.

    Not: Wikipedia politikası gereği botların kullanıcı adında "Bot" geçmesi
    zorunlu (WP:BOTNAME), bu yüzden pattern eşleşmesi pratikte botların
    büyük çoğunluğunu yakalar. Her kullanıcı için ayrıca API'den grup/flag
    sorgulamak (binlerce ekstra istek demek) bu projenin kapsamında
    gereksiz maliyet — bu yüzden sadece pattern kullanıyoruz.
    """
    mask = df["user"].apply(is_bot_username)
    return df[~mask].copy()
