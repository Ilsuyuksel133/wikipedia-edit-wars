"""Revert tespiti: bir revizyonun sha1'i geçmişteki bir revizyonun sha1'i ile
eşleşiyorsa (ve o revizyon aradan sonra geldiyse) revert olarak işaretlenir.
"""


def detect_reverts(revisions_for_page):
    """
    revisions_for_page: tek bir sayfaya ait, zaman sırasına göre sıralı
    revizyon dict listesi (rev_id, sha1, ...).

    Döndürür: {rev_id: reverted_to_rev_id} eşlemesi.

    TODO: sha1 eşleşmesine dayalı identity revert tespiti implement edilecek
    (bkz. Sarvari et al. "wikitrust" / "revert" metodolojisi).
    """
    raise NotImplementedError
