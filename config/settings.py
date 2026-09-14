from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_BASE_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "wikipedia-edit-wars-research/0.1 (educational project)"

# MediaWiki API kibarlık kuralları: saniyede ~1 istekten fazla gitme
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 3

# Sayfa başına çekilecek maksimum revizyon sayısı (None = sınırsız / tüm geçmiş)
MAX_REVISIONS_PER_PAGE = 2000

DB_PATH = PROJECT_ROOT / "data" / "raw" / "wikipedia_edit_wars.db"
