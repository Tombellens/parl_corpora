"""
test_pipeline.py
================
End-to-end test mode: runs the full speaker enrichment pipeline on a small
sample of speakers, using an isolated test database and directories.

Nothing touches the production database.

Usage:
    python3 test_pipeline.py [options]

Options:
    --n N            Number of speakers to test (default: 5)
    --country CC     Filter sample to a specific country (e.g. FR, GB, DE)
    --stage STAGE    Test only one specific stage and stop
                     Choices: query fetch url_synth cv_synth annotate_a annotate_b
                              annotate_c annotate_d
    --keep           Keep the test DB and files after the run (for inspection)
    --verbose        Print LLM inputs/outputs and synthesised text

Example:
    python3 test_pipeline.py --n 3 --country FR --verbose
    python3 test_pipeline.py --n 10 --stage query
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Patch config BEFORE importing anything else that reads config
# (DB_PATH, RAW_HTML_DIR, CV_DIR are all overridden to test locations)
# ---------------------------------------------------------------------------
import config

_TEST_ROOT    = Path(config.ENRICHMENT_DIR) / "_test"
_TEST_DB      = str(_TEST_ROOT / "test.db")
_TEST_HTML    = str(_TEST_ROOT / "raw_html")
_TEST_CV      = str(_TEST_ROOT / "cvs")
_TEST_LOCK    = str(_TEST_ROOT / "llm.lock")

# Monkeypatch — must happen before any other local import
config.DB_PATH       = _TEST_DB
config.RAW_HTML_DIR  = _TEST_HTML
config.CV_DIR        = _TEST_CV
config.LLM_LOCK_FILE = _TEST_LOCK
config.ENRICHMENT_DIR = str(_TEST_ROOT)

# Small batches for test mode
config.BATCH_SIZE_QUERY          = 999
config.BATCH_SIZE_FETCH          = 999
config.BATCH_SIZE_SYNTHESIZE_URL = 999
config.BATCH_SIZE_SYNTHESIZE_CV  = 999
config.BATCH_SIZE_ANNOTATE       = 999

# Now safe to import the rest
from db import (
    FAILED, PENDING, SUCCESS, SKIPPED,
    get_conn, get_stage_counts, init_db, now_iso,
    set_speaker_status, upsert_speaker_url, save_cv,
    save_annotation, get_synthesised_snippets,
)
from llm_client import (
    acquire_llm_lock, chat, extract_json,
    is_llm_locked, load_model, release_llm_lock,
)
from web_cleaner import fetch_and_clean

import batch_query       as _bq
import batch_fetch       as _bf
import batch_synthesize_url as _bsu
import batch_synthesize_cv  as _bscv
import batch_annotate_a  as _ba_a
import batch_annotate_b  as _ba_b
import batch_annotate_c  as _ba_c
import batch_annotate_d  as _ba_d

from import_speakers import make_speaker_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP  = "─" * 70
SEP2 = "═" * 70

def header(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(f"{SEP2}")

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(f"{SEP}")

def ok(msg: str)   -> None: print(f"  ✓ {msg}")
def warn(msg: str) -> None: print(f"  ⚠ {msg}")
def fail(msg: str) -> None: print(f"  ✗ {msg}")
def info(msg: str) -> None: print(f"    {msg}")


def status_line(label: str, value: str, width: int = 20) -> None:
    print(f"  {label:<{width}} {value}")


def print_stage_summary(conn) -> None:
    counts = get_stage_counts(conn)
    print()
    for stage, c in counts.items():
        total   = sum(c.values())
        success = c.get("success", 0)
        failed  = c.get("failed",  0)
        skipped = c.get("skipped", 0)
        pending = c.get("pending", 0)
        bar = (f"  {stage:<14} "
               f"success={success:<4} failed={failed:<4} "
               f"skipped={skipped:<4} pending={pending:<4} / {total}")
        print(bar)


# ---------------------------------------------------------------------------
# Sample loading (direct, without calling import_speakers.main())
# ---------------------------------------------------------------------------

def load_sample(n: int, country: str | None) -> list[dict]:
    """Pull n speakers from speaker_names.csv into the test DB."""
    print(f"  Reading {config.SPEAKER_NAMES_FILE}...")
    df = pd.read_csv(config.SPEAKER_NAMES_FILE, dtype=str)
    persons = df[df["is_person"].str.lower() == "true"].copy()

    if country:
        persons = persons[persons["country"] == country.upper()]
        if persons.empty:
            print(f"  No persons found for country={country}")
            sys.exit(1)

    sample = persons.sample(min(n, len(persons)), random_state=42)

    now = now_iso()
    rows = []
    speaker_dicts = []
    for _, row in sample.iterrows():
        sid = make_speaker_id(
            str(row.get("name_cleaned", "") or ""),
            str(row.get("country",       "") or ""),
            str(row.get("source_dataset","") or ""),
        )
        rows.append((
            sid,
            row.get("name_cleaned") or None,
            row.get("country")      or None,
            row.get("source_dataset") or None,
            row.get("source_dataset_type") or None,
            row.get("min_date") or None,
            row.get("max_date") or None,
            int(row["n_sentences"]) if str(row.get("n_sentences","")).isdigit() else None,
            now, now,
        ))
        speaker_dicts.append({"speaker_id": sid, **row.to_dict()})

    with get_conn() as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO speakers
               (speaker_id, name_cleaned, country, source_dataset, source_dataset_type,
                min_date, max_date, n_sentences, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        total = conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]

    print(f"  Loaded {total} test speakers.")
    return speaker_dicts


def get_test_speakers() -> list:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM speakers").fetchall()


# ---------------------------------------------------------------------------
# Stage runners (thin wrappers that call batch module functions directly,
# with verbose output and per-speaker reporting)
# ---------------------------------------------------------------------------

def run_query(speakers, verbose: bool) -> None:
    section("STAGE: query  (Brave Search API)")
    if not config.BRAVE_API_KEY:
        warn("BRAVE_API_KEY not set — skipping live Brave calls.")
        warn("Inserting 2 dummy URLs per speaker so downstream stages can run.")
        with get_conn() as conn:
            for sp in speakers:
                sid  = sp["speaker_id"]
                name = sp["name_cleaned"] or "Test"
                # Insert dummy Wikipedia-style URLs for testing
                slug = name.lower().replace(" ", "_")
                for i, (lang, url) in enumerate([
                    ("en", f"https://en.wikipedia.org/wiki/{slug}"),
                    ("en", f"https://www.wikidata.org/wiki/Q{abs(hash(name)) % 100000}"),
                ], start=1):
                    upsert_speaker_url(conn, sid, url,
                                       query_language=lang,
                                       query_string=f"[dummy] {name} parliament biography",
                                       search_rank=i,
                                       discovered_at=now_iso())
                set_speaker_status(conn, sid, "query", SUCCESS, query_n_urls=2)
        ok("Dummy URLs inserted.")
        return

    with get_conn() as conn:
        for sp in speakers:
            sid  = sp["speaker_id"]
            name = sp["name_cleaned"] or ""
            info(f"Querying: {name} ({sp['country']})")
            set_speaker_status(conn, sid, "query", "running")
            try:
                ok_result = _bq.process_speaker(conn, dict(sp), run_id="test")
                urls = conn.execute(
                    "SELECT COUNT(*) FROM speaker_urls WHERE speaker_id=?", (sid,)
                ).fetchone()[0]
                ok(f"{name}  →  {urls} unique URLs found")
                if verbose:
                    rows = conn.execute(
                        "SELECT url, query_language, search_rank FROM speaker_urls "
                        "WHERE speaker_id=? ORDER BY search_rank", (sid,)
                    ).fetchall()
                    for r in rows:
                        info(f"  [{r['query_language']}] #{r['search_rank']}  {r['url']}")
            except Exception as e:
                fail(f"{name}: {e}")


def run_fetch(speakers, verbose: bool) -> None:
    section("STAGE: fetch  (web pages)")
    with get_conn() as conn:
        for sp in speakers:
            sid  = sp["speaker_id"]
            name = sp["name_cleaned"] or ""
            info(f"Fetching: {name}")
            pending = conn.execute(
                "SELECT * FROM speaker_urls WHERE speaker_id=? AND fetch_status='pending'",
                (sid,),
            ).fetchall()
            n_ok = n_fail = 0
            for url_row in pending:
                result = fetch_and_clean(url_row["url"], store_raw=True)
                if result.error:
                    conn.execute(
                        "UPDATE speaker_urls SET fetch_status=?, fetch_error=?, fetch_at=? WHERE id=?",
                        (FAILED, result.error, now_iso(), url_row["id"]),
                    )
                    if verbose:
                        warn(f"  FAIL  {url_row['url'][:60]}  ({result.error})")
                    n_fail += 1
                else:
                    conn.execute(
                        """UPDATE speaker_urls
                           SET fetch_status=?, fetch_at=?, fetch_http_status=?,
                               raw_html_path=?, cleaned_text=?, cleaned_text_len=?
                           WHERE id=?""",
                        (SUCCESS, now_iso(), result.http_status,
                         result.raw_html_path, result.cleaned_text,
                         result.cleaned_text_len, url_row["id"]),
                    )
                    if verbose:
                        info(f"  OK    {url_row['url'][:60]}  ({result.cleaned_text_len} chars)")
                    n_ok += 1

            if n_ok > 0:
                set_speaker_status(conn, sid, "fetch", SUCCESS,
                                   fetch_n_success=n_ok, fetch_n_failed=n_fail)
                ok(f"{name}  →  {n_ok} fetched, {n_fail} failed")
            else:
                set_speaker_status(conn, sid, "fetch", FAILED,
                                   error="all URLs failed",
                                   fetch_n_success=0, fetch_n_failed=n_fail)
                fail(f"{name}  →  all {n_fail} URLs failed")


def run_url_synth(speakers, verbose: bool) -> None:
    section("STAGE: url_synth  (per-URL LLM synthesis)")
    acquire_llm_lock("test_url_synth", config.MODEL_SYNTHESIZE_URL)
    try:
        load_model(config.MODEL_SYNTHESIZE_URL)
        with get_conn() as conn:
            for sp in speakers:
                sid  = sp["speaker_id"]
                name = sp["name_cleaned"] or ""
                info(f"Synthesising URLs: {name}")
                set_speaker_status(conn, sid, "url_synth", "running")
                n_done, n_fail = _bsu.process_speaker(conn, dict(sp))
                ok(f"{name}  →  {n_done} snippets, {n_fail} failed")
                if verbose and n_done > 0:
                    snippets = get_synthesised_snippets(conn, sid)
                    for s in snippets:
                        info(f"\n  [source {s['search_rank']}] {s['url'][:60]}")
                        for line in (s["synthesis_text"] or "").splitlines()[:6]:
                            info(f"    {line}")
                        if len((s["synthesis_text"] or "").splitlines()) > 6:
                            info("    …")
    finally:
        release_llm_lock()


def run_cv_synth(speakers, verbose: bool) -> None:
    section("STAGE: cv_synth  (merge snippets → CV)")
    acquire_llm_lock("test_cv_synth", config.MODEL_SYNTHESIZE_CV)
    try:
        load_model(config.MODEL_SYNTHESIZE_CV)
        with get_conn() as conn:
            for sp in speakers:
                sid  = sp["speaker_id"]
                name = sp["name_cleaned"] or ""
                info(f"Merging CV: {name}")
                set_speaker_status(conn, sid, "cv_synth", "running")
                _bscv.process_speaker(conn, dict(sp))
                cv_row = conn.execute(
                    "SELECT cv_text, n_sources_used FROM speaker_cvs WHERE speaker_id=?",
                    (sid,),
                ).fetchone()
                if cv_row:
                    ok(f"{name}  →  CV ({len(cv_row['cv_text'])} chars, "
                       f"{cv_row['n_sources_used']} sources)")
                    if verbose:
                        for line in cv_row["cv_text"].splitlines()[:10]:
                            info(f"  {line}")
                        if len(cv_row["cv_text"].splitlines()) > 10:
                            info("  …")
                else:
                    fail(f"{name}  →  no CV produced")
    finally:
        release_llm_lock()


def _run_annotation(speakers, module, group: str, verbose: bool) -> None:
    acquire_llm_lock(f"test_annotate_{group}", getattr(config, f"MODEL_ANNOTATE_{group}"))
    try:
        load_model(getattr(config, f"MODEL_ANNOTATE_{group}"))
        with get_conn() as conn:
            for sp in speakers:
                sid  = sp["speaker_id"]
                name = sp["name_cleaned"] or ""
                cv_row = conn.execute(
                    "SELECT cv_text, created_at FROM speaker_cvs WHERE speaker_id=?",
                    (sid,),
                ).fetchone()
                if not cv_row or not cv_row["cv_text"]:
                    warn(f"{name}  →  no CV, skipping group {group}")
                    set_speaker_status(conn, sid, f"annotate_{group.lower()}", SKIPPED)
                    continue
                info(f"Annotating [{group}]: {name}")
                set_speaker_status(conn, sid, f"annotate_{group.lower()}", "running")
                try:
                    result = module.annotate(cv_row["cv_text"], name)
                    save_annotation(conn, sid, group, result, SUCCESS,
                                    getattr(config, f"MODEL_ANNOTATE_{group}"),
                                    getattr(config, f"PROMPT_VERSION_ANNOTATE_{group}"),
                                    cv_created_at=cv_row["created_at"])
                    set_speaker_status(conn, sid, f"annotate_{group.lower()}", SUCCESS)
                    ok(f"{name}  →  annotated")
                    if verbose:
                        info(json.dumps(result, indent=4, ensure_ascii=False))
                except Exception as e:
                    save_annotation(conn, sid, group, None, FAILED,
                                    getattr(config, f"MODEL_ANNOTATE_{group}"),
                                    getattr(config, f"PROMPT_VERSION_ANNOTATE_{group}"),
                                    error=str(e))
                    set_speaker_status(conn, sid, f"annotate_{group.lower()}", FAILED,
                                       error=str(e)[:200])
                    fail(f"{name}: {e}")
    finally:
        release_llm_lock()


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_final_report(speakers, verbose: bool) -> None:
    header("TEST RUN COMPLETE — SUMMARY")
    with get_conn() as conn:
        print_stage_summary(conn)
        print()

        for sp in speakers:
            sid  = sp["speaker_id"]
            row  = conn.execute("SELECT * FROM speakers WHERE speaker_id=?", (sid,)).fetchone()
            print(f"\n  {row['name_cleaned']}  ({row['country']} / {row['source_dataset']})")
            stages = [
                ("query",      row["query_status"]),
                ("fetch",      row["fetch_status"]),
                ("url_synth",  row["url_synth_status"]),
                ("cv_synth",   row["cv_synth_status"]),
                ("annotate_a", row["annotate_a_status"]),
                ("annotate_b", row["annotate_b_status"]),
                ("annotate_c", row["annotate_c_status"]),
                ("annotate_d", row["annotate_d_status"]),
            ]
            for stage, status in stages:
                symbol = {"success": "✓", "failed": "✗", "skipped": "–",
                          "pending": "·", "running": "?"}.get(status, "?")
                print(f"    {symbol} {stage:<14} {status}")

            if verbose:
                # Show annotation JSONs
                anns = conn.execute(
                    "SELECT group_name, annotation_json FROM speaker_annotations "
                    "WHERE speaker_id=? AND status='success' ORDER BY group_name",
                    (sid,),
                ).fetchall()
                for ann in anns:
                    try:
                        parsed = json.loads(ann["annotation_json"])
                        print(f"\n    Group {ann['group_name']}:")
                        print("    " + json.dumps(parsed, indent=6,
                              ensure_ascii=False).replace("\n", "\n    "))
                    except Exception:
                        pass

    print(f"\n  Test DB:  {_TEST_DB}")
    print(f"  Raw HTML: {_TEST_HTML}")
    print(f"  CVs:      {_TEST_CV}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGE_ORDER = ["query", "fetch", "url_synth", "cv_synth",
               "annotate_a", "annotate_b", "annotate_c", "annotate_d"]

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline test on a small sample of speakers."
    )
    parser.add_argument("--n",       type=int, default=5,
                        help="Number of speakers to sample (default: 5)")
    parser.add_argument("--country", type=str, default=None,
                        help="Filter sample to one country (ISO-2, e.g. FR)")
    parser.add_argument("--stage",   type=str, default=None,
                        choices=STAGE_ORDER,
                        help="Run only one stage then stop")
    parser.add_argument("--keep",    action="store_true",
                        help="Keep test DB/files after the run")
    parser.add_argument("--verbose", action="store_true",
                        help="Print LLM outputs and per-URL details")
    args = parser.parse_args()

    header(f"SPEAKER ENRICHMENT — TEST MODE"
           f"  (n={args.n}"
           + (f", country={args.country}" if args.country else "")
           + (f", stage={args.stage}" if args.stage else "")
           + ")")

    # Setup isolated test environment
    _TEST_ROOT.mkdir(parents=True, exist_ok=True)
    Path(_TEST_HTML).mkdir(parents=True, exist_ok=True)
    Path(_TEST_CV).mkdir(parents=True, exist_ok=True)

    section("Setup")
    info(f"Test DB:   {_TEST_DB}")
    info(f"Raw HTML:  {_TEST_HTML}")
    info(f"CVs:       {_TEST_CV}")
    init_db()

    section("Loading sample speakers")
    speakers = load_sample(args.n, args.country)
    for sp in speakers:
        info(f"  • {sp.get('name_cleaned','?'):<40} {sp.get('country','?')}  {sp.get('source_dataset','?')}")

    # Run stages
    stages_to_run = [args.stage] if args.stage else STAGE_ORDER

    stage_fns = {
        "query":      lambda: run_query(speakers, args.verbose),
        "fetch":      lambda: run_fetch(speakers, args.verbose),
        "url_synth":  lambda: run_url_synth(speakers, args.verbose),
        "cv_synth":   lambda: run_cv_synth(speakers, args.verbose),
        "annotate_a": lambda: _run_annotation(speakers, _ba_a, "A", args.verbose),
        "annotate_b": lambda: _run_annotation(speakers, _ba_b, "B", args.verbose),
        "annotate_c": lambda: _run_annotation(speakers, _ba_c, "C", args.verbose),
        "annotate_d": lambda: _run_annotation(speakers, _ba_d, "D", args.verbose),
    }

    t_start = time.time()
    for stage in stages_to_run:
        stage_fns[stage]()

    print_final_report(speakers, args.verbose)
    print(f"  Total elapsed: {time.time() - t_start:.1f}s\n")

    if not args.keep:
        print(f"  Cleaning up test files (use --keep to retain them)...")
        shutil.rmtree(str(_TEST_ROOT), ignore_errors=True)
        ok("Test directory removed.")
    else:
        ok("Test files kept for inspection.")


if __name__ == "__main__":
    main()
