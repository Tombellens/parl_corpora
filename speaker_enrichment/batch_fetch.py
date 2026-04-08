"""
batch_fetch.py
==============
Submodule 1, Part 2 — Web page fetching and cleaning.

For each speaker whose query_status=success, fetch all pending URLs:
  - Downloads raw HTML → stores to disk (RAW_HTML_DIR/<sha1>.html)
  - Cleans to plain text via trafilatura / BeautifulSoup
  - Stores cleaned_text in speaker_urls row
  - Updates fetch_status per URL and per speaker (aggregate)

No LLM involved — can run in parallel with query batch if desired.

Usage:
    python3 batch_fetch.py [--limit N] [--failure-batch-id ID]
    nohup python3 batch_fetch.py >> /home/tom/data/speaker_enrichment/logs/fetch.log 2>&1 &
"""

import argparse
import uuid

from tqdm import tqdm

import config
from db import (
    FAILED, PENDING, SUCCESS, SKIPPED,
    activate_failure_batch, create_failure_batch,
    fetch_pending_speakers, get_conn, init_db,
    now_iso, set_speaker_status,
)
from web_cleaner import fetch_and_clean


# ---------------------------------------------------------------------------
# Per-URL fetch
# ---------------------------------------------------------------------------

def fetch_url_row(conn, url_row: dict, ua_index: int = 0) -> bool:
    """Fetch one URL, update its row. Returns True if successful."""
    url_id = url_row["id"]
    url    = url_row["url"]

    result = fetch_and_clean(url, ua_index=ua_index, store_raw=True)

    if result.error:
        conn.execute(
            """UPDATE speaker_urls
               SET fetch_status=?, fetch_error=?, fetch_at=?, fetch_http_status=?
               WHERE id=?""",
            (FAILED, result.error[:500], now_iso(), result.http_status, url_id),
        )
        return False

    conn.execute(
        """UPDATE speaker_urls
           SET fetch_status=?, fetch_at=?, fetch_http_status=?,
               raw_html_path=?, cleaned_text=?, cleaned_text_len=?
           WHERE id=?""",
        (SUCCESS, now_iso(), result.http_status,
         result.raw_html_path, result.cleaned_text, result.cleaned_text_len,
         url_id),
    )
    return True


# ---------------------------------------------------------------------------
# Per-speaker processing
# ---------------------------------------------------------------------------

def process_speaker(conn, speaker: dict) -> tuple[int, int]:
    """
    Fetch all pending URLs for this speaker.
    Returns (n_success, n_failed).
    """
    sid = speaker["speaker_id"]

    pending_urls = conn.execute(
        "SELECT * FROM speaker_urls WHERE speaker_id = ? AND fetch_status = ?",
        (sid, PENDING),
    ).fetchall()

    if not pending_urls:
        return 0, 0

    n_success = n_failed = 0
    for i, url_row in enumerate(pending_urls):
        ok = fetch_url_row(conn, url_row, ua_index=i)
        if ok:
            n_success += 1
        else:
            n_failed += 1

    # Update speaker-level aggregate
    total_success = conn.execute(
        "SELECT COUNT(*) FROM speaker_urls WHERE speaker_id=? AND fetch_status='success'",
        (sid,),
    ).fetchone()[0]
    total_failed = conn.execute(
        "SELECT COUNT(*) FROM speaker_urls WHERE speaker_id=? AND fetch_status='failed'",
        (sid,),
    ).fetchone()[0]

    if total_success == 0:
        status = FAILED
        error  = "all URLs failed to fetch"
    elif total_failed > 0:
        status = SUCCESS   # partial success is still success — some URLs usable
        error  = None
    else:
        status = SUCCESS
        error  = None

    set_speaker_status(conn, sid, "fetch", status, error=error,
                       fetch_n_success=total_success,
                       fetch_n_failed=total_failed)
    return n_success, n_failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=config.BATCH_SIZE_FETCH,
                        help="Max speakers to process this run")
    parser.add_argument("--failure-batch-id", type=int, default=None)
    args = parser.parse_args()

    init_db()
    run_id = str(uuid.uuid4())

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO batch_runs (run_id, stage, batch_type, started_at) VALUES (?,?,?,?)",
            (run_id, "fetch",
             "failure_retry" if args.failure_batch_id else "normal",
             now_iso()),
        )

    # Fetch speakers whose queries are done but pages not yet fetched
    with get_conn() as conn:
        if args.failure_batch_id:
            failure_ids = activate_failure_batch(conn, args.failure_batch_id)
            speakers    = fetch_pending_speakers(conn, "fetch", args.limit,
                                                 failure_ids=failure_ids)
        else:
            speakers = conn.execute(
                """SELECT * FROM speakers
                   WHERE query_status = 'success' AND fetch_status = 'pending'
                   LIMIT ?""",
                (args.limit,),
            ).fetchall()

    print(f"Run {run_id[:8]}  |  {len(speakers)} speakers to fetch")

    n_success = n_failed = total_urls_ok = total_urls_fail = 0
    failed_speaker_ids: list[str] = []

    for speaker in tqdm(speakers, desc="Fetching"):
        sid = speaker["speaker_id"]
        with get_conn() as conn:
            set_speaker_status(conn, sid, "fetch", "running")

        try:
            with get_conn() as conn:
                ok, fail = process_speaker(conn, speaker)
            total_urls_ok   += ok
            total_urls_fail += fail
            n_success += 1
        except Exception as e:
            with get_conn() as conn:
                set_speaker_status(conn, sid, "fetch", FAILED, error=str(e)[:500])
            failed_speaker_ids.append(sid)
            n_failed += 1
            tqdm.write(f"  FAILED {speaker['name_cleaned']}: {e}")

    if failed_speaker_ids:
        with get_conn() as conn:
            fb_id = create_failure_batch(
                conn, "fetch", failed_speaker_ids,
                name=f"fetch_failures_{run_id[:8]}",
                notes=f"Auto-created from run {run_id}",
            )
        print(f"\n  {n_failed} failures quarantined → failure_batch id={fb_id}")

    with get_conn() as conn:
        conn.execute(
            """UPDATE batch_runs
               SET finished_at=?, n_attempted=?, n_success=?, n_failed=?
               WHERE run_id=?""",
            (now_iso(), len(speakers), n_success, n_failed, run_id),
        )

    print(f"\nDone.  speakers: success={n_success} failed={n_failed}")
    print(f"       URLs:     ok={total_urls_ok} failed={total_urls_fail}")


if __name__ == "__main__":
    main()
