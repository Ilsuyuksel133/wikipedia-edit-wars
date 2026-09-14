"""MediaWiki API üzerinden sayfa revizyonlarını çeken ince istemci."""

import time

import requests

from config.settings import API_BASE_URL, MAX_RETRIES, REQUEST_DELAY_SECONDS, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}


def fetch_page_revisions(title, max_revisions=None):
    """
    Bir Wikipedia sayfasının tüm revizyon geçmişini (en eskiden en yeniye)
    sayfalayarak (rvcontinue) çeker.

    Yield eder: dict per revizyon (revid, parentid, user, userid, anon,
    timestamp, size, comment, minor, sha1, tags).
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "redirects": 1,  # "Armenian Genocide" gibi yönlendirme sayfalarını gerçek makaleye çözer
        "rvlimit": 500,
        "rvprop": "ids|timestamp|user|userid|comment|size|flags|sha1|tags",
        # "older": en YENİ revizyondan geriye doğru gidiyoruz. max_revisions
        # sınırına çoğu sayfa çarptığı için (bkz. Faz 2'deki bulgu), "newer"
        # kullanırsak kalabalık sayfaların sadece en ESKİ döneminin verisini
        # alırdık — edit war analizi için asıl önemli olan GÜNCEL dönem.
        "rvdir": "older",
    }

    fetched = 0
    continue_token = None

    while True:
        req_params = dict(params)
        if continue_token:
            req_params["rvcontinue"] = continue_token

        data = _get_with_retry(session, req_params)

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if "missing" in page:
                return
            for rev in page.get("revisions", []):
                rev["page_id"] = int(page_id)
                rev["page_title"] = page.get("title", title)
                yield rev
                fetched += 1
                if max_revisions and fetched >= max_revisions:
                    return

        if "continue" in data:
            continue_token = data["continue"]["rvcontinue"]
            time.sleep(REQUEST_DELAY_SECONDS)
        else:
            break


def _get_with_retry(session, params):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(API_BASE_URL, params=params, timeout=30)
            if resp.status_code == 429:
                # Sunucu "Retry-After" header'ıyla ne kadar beklememiz gerektiğini
                # söyleyebilir; söylemezse katlanarak artan bir süre bekleriz
                # (exponential backoff: 5s, 10s, 20s, ...).
                wait = int(resp.headers.get("Retry-After", 5 * (2**attempt)))
                time.sleep(wait)
                last_error = requests.HTTPError("429 Too Many Requests")
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(REQUEST_DELAY_SECONDS * (attempt + 1))
    raise RuntimeError(f"API isteği {MAX_RETRIES} denemede başarısız: {last_error}")
