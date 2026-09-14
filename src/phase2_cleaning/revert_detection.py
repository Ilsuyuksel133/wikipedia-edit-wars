"""Revert tespiti: bir revizyonun sha1'i geçmişteki bir revizyonun sha1'i ile
eşleşiyorsa (ve aralarında en az bir revizyon varsa) revert olarak işaretlenir.

Bu yönteme "identity revert detection" denir — Wikipedia araştırmalarında
yaygın kullanılan, sha1 checksum karşılaştırmasına dayalı standart teknik.
"""


def detect_reverts(revisions_for_page):
    """
    revisions_for_page: TEK bir sayfaya ait, zaman sırasına göre ARTAN
    sıralı revizyon dict listesi. Her dict en az 'rev_id' ve 'sha1' içermeli.

    Döndürür: {rev_id: {
        'is_revert': bool,             # bu revizyon bir geri alma mı
        'reverted_to_rev_id': int|None,# hangi eski revizyona geri döndü
        'is_reverted': bool,           # bu revizyonun içeriği sonradan silindi mi
        'reverted_by_rev_id': int|None,# hangi revizyon tarafından silindi
    }}
    """
    result = {
        rev["rev_id"]: {
            "is_revert": False,
            "reverted_to_rev_id": None,
            "is_reverted": False,
            "reverted_by_rev_id": None,
        }
        for rev in revisions_for_page
    }

    last_seen_index = {}  # sha1 -> bu içeriğin en son görüldüğü index

    for i, rev in enumerate(revisions_for_page):
        sha1 = rev.get("sha1")
        if not sha1:
            continue

        if sha1 in last_seen_index:
            match_index = last_seen_index[sha1]
            # match_index < i - 1: aralarında en az bir revizyon var demek,
            # yoksa (ardışık aynı sha1) geri alınan bir şey yok, anlamsız.
            if match_index < i - 1:
                current_id = rev["rev_id"]
                matched_id = revisions_for_page[match_index]["rev_id"]

                result[current_id]["is_revert"] = True
                result[current_id]["reverted_to_rev_id"] = matched_id

                for j in range(match_index + 1, i):
                    reverted_id = revisions_for_page[j]["rev_id"]
                    result[reverted_id]["is_reverted"] = True
                    result[reverted_id]["reverted_by_rev_id"] = current_id

        last_seen_index[sha1] = i

    return result
