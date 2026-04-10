"""
config.py
=========
Central configuration for the speaker enrichment pipeline.
All paths, API settings, language mappings, and batch sizes live here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR        = "/home/tom/data"
ENRICHMENT_DIR  = f"{DATA_DIR}/speaker_enrichment"
DB_PATH         = f"{ENRICHMENT_DIR}/speaker_enrichment.db"
RAW_HTML_DIR    = f"{ENRICHMENT_DIR}/raw_html"       # stores raw HTML as files
CV_DIR          = f"{ENRICHMENT_DIR}/cvs"             # stores CV text files
LOG_DIR         = f"{ENRICHMENT_DIR}/logs"
LLM_LOCK_FILE   = f"{ENRICHMENT_DIR}/llm.lock"

SPEAKER_NAMES_FILE = f"{DATA_DIR}/speaker_names.csv"

# ---------------------------------------------------------------------------
# Brave Search API
# ---------------------------------------------------------------------------
BRAVE_API_KEY         = os.environ.get("BRAVE_API_KEY", "")
BRAVE_ENDPOINT        = "https://api.search.brave.com/res/v1/web/search"
BRAVE_RESULTS_PER_QUERY = 10
BRAVE_RATE_LIMIT_DELAY  = 1.1   # seconds between calls (stay under free-tier limit)
# Max Brave API calls per speaker.  Each call = $0.005.
# Strategy: primary country language + English = at most 2 calls/speaker.
# For English-native countries (GB, AU, US, NZ) this is already 1 call.
# Set to 1 to use only the primary language and cut costs in half.
BRAVE_MAX_QUERIES_PER_SPEAKER = 2

# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------
LM_STUDIO_BASE_URL = "http://localhost:1234"
LM_STUDIO_API_KEY  = "lm-studio"

# Models — fill in once you decide; these are placeholders
MODEL_SYNTHESIZE_URL = "openai/gpt-oss-20b"   # per-URL synthesis
MODEL_SYNTHESIZE_CV  = "openai/gpt-oss-20b"   # CV merger
MODEL_ANNOTATE_A     = "openai/gpt-oss-20b"   # group A
MODEL_ANNOTATE_B     = "openai/gpt-oss-20b"   # group B
MODEL_ANNOTATE_C     = "openai/gpt-oss-20b"   # group C
MODEL_ANNOTATE_D     = "openai/gpt-oss-20b"   # group D

# Prompt versions — bump when you change prompts to keep the DB traceable
PROMPT_VERSION_SYNTHESIZE_URL = "1.0"
PROMPT_VERSION_SYNTHESIZE_CV  = "1.0"
PROMPT_VERSION_ANNOTATE_A     = "1.0"
PROMPT_VERSION_ANNOTATE_B     = "1.0"
PROMPT_VERSION_ANNOTATE_C     = "1.0"
PROMPT_VERSION_ANNOTATE_D     = "1.0"

# ---------------------------------------------------------------------------
# Batch sizes
# ---------------------------------------------------------------------------
BATCH_SIZE_QUERY          = 1000   # persons queried per daily run
BATCH_SIZE_FETCH          = 500    # URLs fetched per run
BATCH_SIZE_SYNTHESIZE_URL = 200    # URLs synthesized per LLM run
BATCH_SIZE_SYNTHESIZE_CV  = 100    # CVs merged per LLM run
BATCH_SIZE_ANNOTATE       = 100    # persons annotated per LLM run (all groups)

# Web fetching
FETCH_TIMEOUT_SECONDS = 20
MAX_CLEANED_TEXT_CHARS = 60_000    # truncation ceiling fed to the LLM

# Dashboard
DASHBOARD_PORT = 5050
DASHBOARD_HOST = "0.0.0.0"        # bind on all interfaces for SSH tunnelling

# ---------------------------------------------------------------------------
# Country → languages (ISO 639-1)
# ---------------------------------------------------------------------------
COUNTRY_LANGUAGES: dict[str, list[str]] = {
    "AT": ["de", "en"],
    "AU": ["en"],
    "BA": ["bs", "en"],
    "BE": ["fr", "nl", "de", "en"],
    "BG": ["bg", "en"],
    "CA": ["en", "fr"],
    "CZ": ["cs", "en"],
    "DE": ["de", "en"],
    "DK": ["da", "en"],
    "EE": ["et", "en"],
    "ES": ["es", "en"],
    "FI": ["fi", "en"],
    "FR": ["fr", "en"],
    "GB": ["en"],
    "GR": ["el", "en"],
    "HR": ["hr", "en"],
    "HU": ["hu", "en"],
    "IS": ["is", "en"],
    "IT": ["it", "en"],
    "LT": ["lt", "en"],
    "LV": ["lv", "en"],
    "NL": ["nl", "en"],
    "NO": ["no", "en"],
    "NZ": ["en"],
    "PL": ["pl", "en"],
    "PT": ["pt", "en"],
    "RO": ["ro", "en"],
    "RS": ["sr", "en"],
    "SE": ["sv", "en"],
    "SI": ["sl", "en"],
    "SK": ["sk", "en"],
    "TR": ["tr", "en"],
    "UA": ["uk", "en"],
    "US": ["en"],
}

# ---------------------------------------------------------------------------
# "parliament" in each query language
# ---------------------------------------------------------------------------
PARLIAMENT_WORD: dict[str, str] = {
    "en": "parliament",
    "de": "Parlament",
    "fr": "parlement",
    "nl": "parlement",
    "it": "parlamento",
    "es": "parlamento",
    "da": "folketing",
    "sv": "riksdag",
    "cs": "parlament",
    "bg": "парламент",
    "bs": "parlament",
    "et": "riigikogu",
    "sk": "parlament",
    "sl": "parlament",
    "pl": "sejm",
    "hr": "sabor",
    "hu": "parlament",
    "el": "βουλή",
    "pt": "parlamento",
    "ro": "parlament",
    "lv": "saeima",
    "lt": "seimas",
    "is": "alþingi",
    "no": "stortinget",
    "fi": "eduskunta",
    "sr": "скупштина",
    "tr": "meclis",
    "uk": "парламент",
}

# ---------------------------------------------------------------------------
# "biography" in each query language
# ---------------------------------------------------------------------------
BIOGRAPHY_WORD: dict[str, str] = {
    "en": "biography",
    "de": "Biografie",
    "fr": "biographie",
    "nl": "biografie",
    "it": "biografia",
    "es": "biografía",
    "da": "biografi",
    "sv": "biografi",
    "cs": "biografie",
    "bg": "биография",
    "bs": "biografija",
    "et": "biograafia",
    "sk": "biografia",
    "sl": "biografija",
    "pl": "biografia",
    "hr": "biografija",
    "hu": "életrajz",
    "el": "βιογραφία",
    "pt": "biografia",
    "ro": "biografie",
    "lv": "biogrāfija",
    "lt": "biografija",
    "is": "ævisaga",
    "no": "biografi",
    "fi": "elämäkerta",
    "sr": "биографија",
    "tr": "biyografi",
    "uk": "біографія",
}
