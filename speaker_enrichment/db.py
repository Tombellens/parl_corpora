"""
db.py
=====
SQLite schema definition and helper functions for the speaker enrichment pipeline.

Every row tracks provenance at the finest useful granularity:
  - speakers        : one row per unique person (is_person=True in speaker_names.csv)
  - speaker_urls    : one row per (speaker × URL) — stores raw HTML path, cleaned text,
                      per-URL synthesis text
  - speaker_cvs     : one row per speaker — merged CV text + list of source URL ids
  - speaker_annotations : one row per (speaker × annotation group A/B/C/D)
  - failure_batches : manually-activated retry collections
  - batch_runs      : audit log for every batch execution

Design goals:
  - Everything stored, nothing discarded (scientific paper traceability)
  - Safe to re-run: all batch scripts check status before doing work
  - Failures quarantined, never auto-retried without human decision
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from config import DB_PATH, ENRICHMENT_DIR

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
PENDING  = "pending"
RUNNING  = "running"
SUCCESS  = "success"
FAILED   = "failed"
SKIPPED  = "skipped"   # e.g. no URLs found → no synthesis possible

ALL_STATUSES = {PENDING, RUNNING, SUCCESS, FAILED, SKIPPED}

STAGES = [
    "query",
    "fetch",
    "url_synth",
    "cv_synth",
    "annotate_a",
    "annotate_b",
    "annotate_c",
    "annotate_d",
]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS speakers (
    speaker_id          TEXT PRIMARY KEY,
    name_cleaned        TEXT,
    country             TEXT,
    source_dataset      TEXT,
    source_dataset_type TEXT,
    min_date            TEXT,
    max_date            TEXT,
    n_sentences         INTEGER,

    -- Stage statuses
    query_status        TEXT NOT NULL DEFAULT 'pending',
    query_error         TEXT,
    query_at            TEXT,
    query_n_urls        INTEGER DEFAULT 0,

    fetch_status        TEXT NOT NULL DEFAULT 'pending',
    fetch_error         TEXT,
    fetch_at            TEXT,
    fetch_n_success     INTEGER DEFAULT 0,
    fetch_n_failed      INTEGER DEFAULT 0,

    url_synth_status    TEXT NOT NULL DEFAULT 'pending',
    url_synth_error     TEXT,
    url_synth_at        TEXT,
    url_synth_model     TEXT,
    url_synth_prompt_v  TEXT,
    url_synth_n_done    INTEGER DEFAULT 0,

    cv_synth_status     TEXT NOT NULL DEFAULT 'pending',
    cv_synth_error      TEXT,
    cv_synth_at         TEXT,
    cv_synth_model      TEXT,
    cv_synth_prompt_v   TEXT,

    annotate_a_status   TEXT NOT NULL DEFAULT 'pending',
    annotate_a_error    TEXT,
    annotate_a_at       TEXT,
    annotate_a_model    TEXT,
    annotate_a_prompt_v TEXT,

    annotate_b_status   TEXT NOT NULL DEFAULT 'pending',
    annotate_b_error    TEXT,
    annotate_b_at       TEXT,
    annotate_b_model    TEXT,
    annotate_b_prompt_v TEXT,

    annotate_c_status   TEXT NOT NULL DEFAULT 'pending',
    annotate_c_error    TEXT,
    annotate_c_at       TEXT,
    annotate_c_model    TEXT,
    annotate_c_prompt_v TEXT,

    annotate_d_status   TEXT NOT NULL DEFAULT 'pending',
    annotate_d_error    TEXT,
    annotate_d_at       TEXT,
    annotate_d_model    TEXT,
    annotate_d_prompt_v TEXT,

    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS speaker_urls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id          TEXT NOT NULL REFERENCES speakers(speaker_id),
    url                 TEXT NOT NULL,
    query_language      TEXT,          -- language code of the query that found this URL
    query_string        TEXT,          -- the exact query string used
    search_rank         INTEGER,       -- position in search results (1-10)
    discovered_at       TEXT,

    fetch_status        TEXT NOT NULL DEFAULT 'pending',
    fetch_error         TEXT,
    fetch_at            TEXT,
    fetch_http_status   INTEGER,
    raw_html_path       TEXT,          -- relative path under RAW_HTML_DIR (for traceability)
    cleaned_text        TEXT,          -- trafilatura-extracted plain text
    cleaned_text_len    INTEGER,

    synthesis_status    TEXT NOT NULL DEFAULT 'pending',
    synthesis_error     TEXT,
    synthesis_at        TEXT,
    synthesis_model     TEXT,
    synthesis_prompt_v  TEXT,
    synthesis_text      TEXT,          -- per-URL LLM synthesis snippet

    UNIQUE(speaker_id, url)
);

CREATE TABLE IF NOT EXISTS speaker_cvs (
    speaker_id          TEXT PRIMARY KEY REFERENCES speakers(speaker_id),
    cv_text             TEXT,
    cv_file_path        TEXT,          -- path under CV_DIR (for traceability)
    source_url_ids      TEXT,          -- JSON array of speaker_urls.id values used
    n_sources_used      INTEGER,
    model               TEXT,
    prompt_version      TEXT,
    created_at          TEXT,
    updated_at          TEXT
);

CREATE TABLE IF NOT EXISTS speaker_annotations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id      TEXT NOT NULL REFERENCES speakers(speaker_id),
    group_name      TEXT NOT NULL,     -- 'A', 'B', 'C', 'D'
    annotation_json TEXT,              -- structured JSON output from LLM
    status          TEXT NOT NULL DEFAULT 'pending',
    error           TEXT,
    annotated_at    TEXT,
    model           TEXT,
    prompt_version  TEXT,
    cv_created_at   TEXT,              -- timestamp of CV used (version reference)
    UNIQUE(speaker_id, group_name)
);

-- Manually-activated failure retry batches
CREATE TABLE IF NOT EXISTS failure_batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    stage           TEXT NOT NULL,
    speaker_ids     TEXT NOT NULL,     -- JSON array of speaker_ids
    n_speakers      INTEGER,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    activated_at    TEXT,              -- NULL until dashboard action
    completed_at    TEXT,
    notes           TEXT
);

-- Audit log: one row per batch script run
CREATE TABLE IF NOT EXISTS batch_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,     -- UUID
    stage           TEXT NOT NULL,
    batch_type      TEXT NOT NULL DEFAULT 'normal',  -- 'normal' | 'failure_retry'
    failure_batch_id INTEGER REFERENCES failure_batches(id),
    started_at      TEXT,
    finished_at     TEXT,
    n_attempted     INTEGER DEFAULT 0,
    n_success       INTEGER DEFAULT 0,
    n_failed        INTEGER DEFAULT 0,
    n_skipped       INTEGER DEFAULT 0,
    notes           TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_speakers_country       ON speakers(country);
CREATE INDEX IF NOT EXISTS idx_speakers_query_status  ON speakers(query_status);
CREATE INDEX IF NOT EXISTS idx_speakers_fetch_status  ON speakers(fetch_status);
CREATE INDEX IF NOT EXISTS idx_speakers_url_synth     ON speakers(url_synth_status);
CREATE INDEX IF NOT EXISTS idx_speakers_cv_synth      ON speakers(cv_synth_status);
CREATE INDEX IF NOT EXISTS idx_speakers_ann_a         ON speakers(annotate_a_status);
CREATE INDEX IF NOT EXISTS idx_speakers_ann_b         ON speakers(annotate_b_status);
CREATE INDEX IF NOT EXISTS idx_speakers_ann_c         ON speakers(annotate_c_status);
CREATE INDEX IF NOT EXISTS idx_speakers_ann_d         ON speakers(annotate_d_status);
CREATE INDEX IF NOT EXISTS idx_urls_speaker           ON speaker_urls(speaker_id);
CREATE INDEX IF NOT EXISTS idx_urls_fetch_status      ON speaker_urls(fetch_status);
CREATE INDEX IF NOT EXISTS idx_urls_synthesis_status  ON speaker_urls(synthesis_status);
"""

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
@contextmanager
def get_conn(db_path: str | None = None):
    if db_path is None:
        import config as _cfg
        db_path = _cfg.DB_PATH
    conn = sqlite3.connect(db_path, timeout=30)
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
    """Create all tables and indexes if they do not exist."""
    import config as _cfg
    if db_path is None:
        db_path = _cfg.DB_PATH
    Path(_cfg.ENRICHMENT_DIR).mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
    print(f"Database initialised at {db_path}")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_speaker_status(conn: sqlite3.Connection, speaker_id: str,
                        stage: str, status: str,
                        error: str | None = None, **extra_cols) -> None:
    """Update a stage status column (and optional extra columns) for one speaker."""
    cols = {f"{stage}_status": status, "updated_at": now_iso()}
    if error is not None:
        cols[f"{stage}_error"] = error
    cols[f"{stage}_at"] = now_iso()
    cols.update(extra_cols)

    set_clause = ", ".join(f"{k} = ?" for k in cols)
    values = list(cols.values()) + [speaker_id]
    conn.execute(f"UPDATE speakers SET {set_clause} WHERE speaker_id = ?", values)


def fetch_pending_speakers(conn: sqlite3.Connection, stage: str,
                            limit: int, failure_ids: list[str] | None = None
                            ) -> list[sqlite3.Row]:
    """Return up to `limit` speakers whose stage status is 'pending'."""
    if failure_ids is not None:
        placeholders = ",".join("?" * len(failure_ids))
        return conn.execute(
            f"SELECT * FROM speakers WHERE speaker_id IN ({placeholders})",
            failure_ids,
        ).fetchall()
    return conn.execute(
        f"SELECT * FROM speakers WHERE {stage}_status = 'pending' LIMIT ?",
        (limit,),
    ).fetchall()


def get_stage_counts(conn: sqlite3.Connection) -> dict:
    """Return {stage: {status: count}} for the dashboard."""
    result = {}
    for stage in STAGES:
        col = f"{stage}_status"
        rows = conn.execute(
            f"SELECT {col}, COUNT(*) FROM speakers GROUP BY {col}"
        ).fetchall()
        result[stage] = {r[0]: r[1] for r in rows}
    return result


def upsert_speaker_url(conn: sqlite3.Connection, speaker_id: str, url: str,
                        **kwargs) -> int:
    """Insert a URL row (ignore if already exists). Returns the row id."""
    cols = {"speaker_id": speaker_id, "url": url, **kwargs}
    col_names = ", ".join(cols.keys())
    placeholders = ", ".join("?" * len(cols))
    conn.execute(
        f"INSERT OR IGNORE INTO speaker_urls ({col_names}) VALUES ({placeholders})",
        list(cols.values()),
    )
    row = conn.execute(
        "SELECT id FROM speaker_urls WHERE speaker_id = ? AND url = ?",
        (speaker_id, url),
    ).fetchone()
    return row["id"]


def get_synthesised_snippets(conn: sqlite3.Connection,
                              speaker_id: str) -> list[sqlite3.Row]:
    """Return all successfully synthesised URL rows for a speaker."""
    return conn.execute(
        """SELECT id, url, query_language, search_rank, synthesis_text
           FROM speaker_urls
           WHERE speaker_id = ? AND synthesis_status = 'success'
           ORDER BY search_rank""",
        (speaker_id,),
    ).fetchall()


def save_cv(conn: sqlite3.Connection, speaker_id: str, cv_text: str,
            cv_file_path: str, source_url_ids: list[int],
            model: str, prompt_version: str) -> None:
    ts = now_iso()
    conn.execute(
        """INSERT INTO speaker_cvs
           (speaker_id, cv_text, cv_file_path, source_url_ids, n_sources_used,
            model, prompt_version, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(speaker_id) DO UPDATE SET
             cv_text=excluded.cv_text,
             cv_file_path=excluded.cv_file_path,
             source_url_ids=excluded.source_url_ids,
             n_sources_used=excluded.n_sources_used,
             model=excluded.model,
             prompt_version=excluded.prompt_version,
             updated_at=excluded.updated_at""",
        (speaker_id, cv_text, cv_file_path,
         json.dumps(source_url_ids), len(source_url_ids),
         model, prompt_version, ts, ts),
    )


def save_annotation(conn: sqlite3.Connection, speaker_id: str,
                    group_name: str, annotation_json: dict | None,
                    status: str, model: str, prompt_version: str,
                    cv_created_at: str | None = None,
                    error: str | None = None) -> None:
    ts = now_iso()
    conn.execute(
        """INSERT INTO speaker_annotations
           (speaker_id, group_name, annotation_json, status, error,
            annotated_at, model, prompt_version, cv_created_at)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(speaker_id, group_name) DO UPDATE SET
             annotation_json=excluded.annotation_json,
             status=excluded.status,
             error=excluded.error,
             annotated_at=excluded.annotated_at,
             model=excluded.model,
             prompt_version=excluded.prompt_version,
             cv_created_at=excluded.cv_created_at""",
        (speaker_id, group_name,
         json.dumps(annotation_json) if annotation_json else None,
         status, error, ts, model, prompt_version, cv_created_at),
    )


# ---------------------------------------------------------------------------
# Failure batch helpers
# ---------------------------------------------------------------------------
def create_failure_batch(conn: sqlite3.Connection, stage: str,
                          speaker_ids: list[str], name: str = "",
                          notes: str = "") -> int:
    cur = conn.execute(
        """INSERT INTO failure_batches (name, stage, speaker_ids, n_speakers, notes)
           VALUES (?,?,?,?,?)""",
        (name, stage, json.dumps(speaker_ids), len(speaker_ids), notes),
    )
    return cur.lastrowid


def activate_failure_batch(conn: sqlite3.Connection, batch_id: int) -> list[str]:
    """Mark a failure batch as activated and reset speaker statuses to pending."""
    row = conn.execute(
        "SELECT stage, speaker_ids FROM failure_batches WHERE id = ?",
        (batch_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Failure batch {batch_id} not found")

    stage = row["stage"]
    speaker_ids = json.loads(row["speaker_ids"])

    # Reset speaker statuses to pending for that stage
    for sid in speaker_ids:
        conn.execute(
            f"UPDATE speakers SET {stage}_status = 'pending', {stage}_error = NULL "
            f"WHERE speaker_id = ?",
            (sid,),
        )

    conn.execute(
        "UPDATE failure_batches SET activated_at = ? WHERE id = ?",
        (now_iso(), batch_id),
    )
    return speaker_ids
