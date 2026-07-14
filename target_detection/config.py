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
LLM_CONTEXT_LENGTH = 8192        # small: accusation + context batches, not full CVs
LLM_NUM_PARALLEL   = 4           # inputs are tiny, so we can afford parallel slots
PROMPT_VERSION     = "1.0"

# Batch sizes
TARGET_BATCH_ACCUSATIONS = 20    # accusations classified per LLM call (throughput)
TARGET_MAX_TOKENS        = 1400  # room for a JSON array of ~20 short results

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
