"""Bot editörlerini kullanıcı adı pattern'leri + MediaWiki bot flag'i ile filtreleme."""

import re

BOT_USERNAME_PATTERNS = [
    re.compile(r"bot$", re.IGNORECASE),
    re.compile(r"^bot", re.IGNORECASE),
]


def is_bot_username(username):
    if not username:
        return False
    return any(p.search(username) for p in BOT_USERNAME_PATTERNS)


def filter_bot_revisions(df):
    """revisions DataFrame'inden bot editörlerine ait satırları çıkarır.

    TODO: MediaWiki API'den user grupları (bot flag) çekilip pattern
    eşleşmesine ek doğrulama olarak kullanılacak.
    """
    raise NotImplementedError
