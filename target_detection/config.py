"""
config.py
=========
Configuration for the accusation target-detection pipeline.

Stage 1 of the "beyond individual-level" work: for each detected accusation
(lie_label == LABEL_1 in the lielines-scored sentence corpus), determine the
TARGET of the accusation — a broad type plus the raw mention. Person targets
are resolved to speakers in a later stage.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR         = "/home/tom/data"
PREDICTED_CSV    = f"{DATA_DIR}/sentence_corpus_predicted.csv"   # lielines output (149M rows)
TARGET_DIR       = f"{DATA_DIR}/target_detection"
DB_PATH          = f"{TARGET_DIR}/accusations.db"

ACCUSATION_LABEL = "LABEL_1"     # lielines label marking a lie/untruth accusation
CONTEXT_WINDOW   = 3             # sentences of context on each side (same speech)

# ---------------------------------------------------------------------------
# LM Studio (reuses the hardened client from ../speaker_enrichment)
# ---------------------------------------------------------------------------
MODEL              = "openai/gpt-oss-20b"
LLM_CONTEXT_LENGTH = 16384       # holds a batch of accusation+context items
LLM_NUM_PARALLEL   = 1           # one batched prompt at a time (crash-recovery safe)
PROMPT_VERSION     = "1.0"

# Constants required by the shared llm_client (../speaker_enrichment/llm_client.py).
# It does `from config import ...`, and because this module's directory is first
# on sys.path these names must live here too. They point at the same LM Studio
# server; the lock file is SHARED with speaker_enrichment so the two pipelines
# cannot both grab the single GPU at once.
LM_STUDIO_BASE_URL         = "http://localhost:1234"
LM_STUDIO_API_KEY          = "lm-studio"
LMS_BIN                    = os.path.expanduser("~/.lmstudio/bin/lms")
LMS_SERVER_STARTUP_TIMEOUT = 60
LLM_LOCK_FILE              = "/home/tom/data/speaker_enrichment/llm.lock"

# Batch sizes — many small accusations per call is the throughput lever
TARGET_BATCH_ACCUSATIONS = 30    # accusations classified per LLM call
TARGET_MAX_TOKENS        = 2200  # room for a JSON array of ~30 short results

# ---------------------------------------------------------------------------
# Target taxonomy (Phase 1). Broad by design.
# ---------------------------------------------------------------------------
TARGET_TYPES = [
    "person",                  # a specific named individual
    "government",              # a government function holder, or the government as a whole
    "administration",          # civil service / bureaucracy / a ministry as apparatus
    "political_party",         # a party or parliamentary group
    "public_institution",      # courts, central bank, agencies, army/police, other state bodies
    "foreign_country_or_govt", # another country or its government
    "international_org",        # EU, UN, NATO, ...
    "media",                   # press, broadcasters, journalists
    "social_group",            # a demographic / social / ethnic / religious group
    "other",                   # a real target that fits none of the above
    "unclear_or_none",         # no identifiable target / purely rhetorical
]
TARGET_TYPE_SET = set(TARGET_TYPES)
