"""
db.py
=====
SQLite schema + helpers for the accusation target-detection pipeline.

One row per detected accusation (lielines LABEL_1), carrying its context window
and, once processed, the detected target. Resumable and provenance-tracked, in
the same spirit as the speaker_enrichment DB.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import config

PENDING = "pending"
RUNNING = "running"
SUCCESS = "success"
FAILED  = "failed"

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS accusations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- provenance back into the sentence corpus
    source_dataset      TEXT,
    source_dataset_type TEXT,
    source_file         TEXT,
    source_speech_id    TEXT,
    sentence_idx        INTEGER,

    date                TEXT,
    speaker             TEXT,
    country             TEXT,

    sentence            TEXT NOT NULL,   -- the accusation sentence
    context             TEXT,            -- +/- CONTEXT_WINDOW sentences, accusation marked
    lie_score           REAL,

    -- target detection (Phase 1)
    target_status       TEXT NOT NULL DEFAULT 'pending',
    target_type         TEXT,
    target_text         TEXT,            -- raw mention as written
    target_error        TEXT,
    target_at           TEXT,
    target_model        TEXT,
    target_prompt_v     TEXT,

    -- person resolution (Phase 2, filled later for target_type='person')
    resolved_speaker_id TEXT,
    resolve_status      TEXT NOT NULL DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_acc_target_status ON accusations(target_status);
CREATE INDEX IF NOT EXISTS idx_acc_country       ON accusations(country);
CREATE INDEX IF NOT EXISTS idx_acc_target_type   ON accusations(target_type);
CREATE INDEX IF NOT EXISTS idx_acc_resolve       ON accusations(resolve_status);
"""


@contextmanager
def get_conn(db_path: str | None = None):
    conn = sqlite3.connect(db_path or config.DB_PATH, timeout=300)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    path = db_path or config.DB_PATH
    Path(config.TARGET_DIR).mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
    print(f"Database initialised at {path}")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
