"""
batch_query.py
==============
Submodule 1, Part 1 — Brave Search API querying.

For each pending speaker:
  - Builds one query per country language: "{name}" {parliament_word} {biography_word}
  - Always includes an English query
  - Fetches top-10 results per query from the Brave Search API
  - Deduplicates URLs across all queries for that speaker
  - Inserts URL rows into speaker_urls (with query provenance)
  - Updates speaker.query_status

Run daily. Processes BATCH_SIZE_QUERY speakers per run.
Failures are recorded and quarantined into a failure_batch for manual review.

Usage:
    python3 batch_query.py [--limit N] [--failure-batch-id ID]
    nohup python3 batch_query.py >> /home/tom/data/speaker_enrichment/logs/query.log 2>&1 &
"""

import argparse
import time
import uuid
from urllib.parse import urlparse

import requests
from tqdm import tqdm

import config
from db import (
    FAILED, PENDING, SUCCESS,
    activate_failure_batch, create_failure_batch,
    fetch_pending_speakers, get_conn, init_db,
    now_iso, set_speaker_status, upsert_speaker_url,
)


# ---------------------------------------------------------------------------
# Domain blacklist — sites that will never yield scrapable biographical text
# ---------------------------------------------------------------------------
BLACKLISTED_DOMAINS = {
    # Social media / login-walled
    "facebook.com", "www.facebook.com",
    "instagram.com", "www.instagram.com",
    "twitter.com", "www.twitter.com", "x.com", "www.x.com",
    "linkedin.com", "www.linkedin.com",
    "tiktok.com", "www.tiktok.com",
    "youtube.com", "www.youtube.com",
    # Image / media asset hosts (no text)
    "bilddatenbank.bundestag.de",
    # Generic homepages / search engines
    "google.com", "www.google.com",
    "bing.com", "www.bing.com",
    "duckduckgo.com",
}


def _is_blacklisted(url: str) -> bool:
    """Return True if the URL's domain is in the blacklist."""
    try:
        host = urlparse(url).hostname or ""
        return host in BLACKLISTED_DOMAINS or any(
            host.endswith("." + d) for d in BLACKLISTED_DOMAINS
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Brave Search
# ---------------------------------------------------------------------------

def brave_search(query: str, count: int = 10) -> list[dict]:
    """
    Call the Brave Web Search API and return up to `count` result dicts,
    each with keys: url, title, description.
    Returns [] on any error.
    """
    if not config.BRAVE_API_KEY:
        raise RuntimeError("BRAVE_API_KEY environment variable is not set.")

    headers = {
        "Accept":                "application/json",
        "Accept-Encoding":       "gzip",
        "X-Subscription-Token":  config.BRAVE_API_KEY,
    }
    params = {"q": query, "count": count, "text_decorations": "false"}

    try:
        resp = requests.get(
            config.BRAVE_ENDPOINT,
            headers=headers,
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("web", {}).get("results", [])
        return [
            {
                "url":   r.get("url", ""),
                "title": r.get("title", ""),
                "desc":  r.get("description", ""),
            }
            for r in results
            if r.get("url")
        ]
    except Exception as e:
        print(f"    Brave API error for '{query}': {e}")
        return []


def build_queries(name: str, country: str) -> list[tuple[str, str]]:
    """
    Return [(query_string, language_code), ...] for this speaker.

    Strategy (cost-aware):
      - Always query in the primary country language (index 0).
      - Always add an English query if the primary language is not English.
      - Never exceed BRAVE_MAX_QUERIES_PER_SPEAKER API calls.

    This caps Belgium (fr/nl/de/en) at 2 calls (fr + en) instead of 4,
    keeping per-speaker cost at ≤$0.01 regardless of country.
    """
    langs = config.COUNTRY_LANGUAGES.get(country, ["en"])
    primary = langs[0]

    # Build the priority list: primary language first, then English (if different)
    priority = [primary]
    if "en" not in priority:
        priority.append("en")

    # Respect the hard cap
    priority = priority[: config.BRAVE_MAX_QUERIES_PER_SPEAKER]

    queries = []
    for lang in priority:
        parl = config.PARLIAMENT_WORD.get(lang, "parliament")
        bio  = config.BIOGRAPHY_WORD.get(lang, "biography")
        q = f'"{name}" {parl} {bio}'
        queries.append((q, lang))
    return queries


# ---------------------------------------------------------------------------
# Per-speaker processing
# ---------------------------------------------------------------------------

def process_speaker(conn, speaker: dict, run_id: str) -> bool:
    """
    Query Brave for this speaker, insert URL rows.
    Returns True on success, False on failure.
    """
    sid   = speaker["speaker_id"]
    name  = speaker["name_cleaned"] or ""
    country = speaker["country"] or "GB"

    if not name.strip():
        set_speaker_status(conn, sid, "query", "skipped",
                           error="empty name_cleaned")
        return True

    queries = build_queries(name, country)
    seen_urls: set[str] = set()
    total_urls = 0
    discovered_at = now_iso()

    for q_string, lang in queries:
        results = brave_search(q_string, count=config.BRAVE_RESULTS_PER_QUERY)
        time.sleep(config.BRAVE_RATE_LIMIT_DELAY)

        for rank, r in enumerate(results, start=1):
            url = r["url"].strip()
            if not url or url in seen_urls:
                continue
            if _is_blacklisted(url):
                continue
            seen_urls.add(url)
            upsert_speaker_url(
                conn, sid, url,
                query_language=lang,
                query_string=q_string,
                search_rank=rank,
                discovered_at=discovered_at,
            )
            total_urls += 1

    set_speaker_status(conn, sid, "query", SUCCESS, query_n_urls=total_urls)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=config.BATCH_SIZE_QUERY,
                        help="Max speakers to process this run")
    parser.add_argument("--failure-batch-id", type=int, default=None,
                        help="Activate and process a specific failure batch")
    args = parser.parse_args()

    init_db()
    run_id = str(uuid.uuid4())
    failed_ids: list[str] = []

    with get_conn() as conn:
        # Log run start
        conn.execute(
            "INSERT INTO batch_runs (run_id, stage, batch_type, started_at) "
            "VALUES (?,?,?,?)",
            (run_id, "query",
             "failure_retry" if args.failure_batch_id else "normal",
             now_iso()),
        )

        # Fetch speakers to process
        if args.failure_batch_id:
            failure_ids = activate_failure_batch(conn, args.failure_batch_id)
            speakers    = fetch_pending_speakers(conn, "query", args.limit,
                                                 failure_ids=failure_ids)
        else:
            speakers = fetch_pending_speakers(conn, "query", args.limit)

    print(f"Run {run_id[:8]}  |  {len(speakers)} speakers to query")

    n_success = n_failed = n_skipped = 0

    for speaker in tqdm(speakers, desc="Querying"):
        sid = speaker["speaker_id"]
        with get_conn() as conn:
            set_speaker_status(conn, sid, "query", "running")

        try:
            with get_conn() as conn:
                ok = process_speaker(conn, speaker, run_id)
            if ok:
                n_success += 1
            else:
                n_skipped += 1
        except Exception as e:
            with get_conn() as conn:
                set_speaker_status(conn, sid, "query", FAILED,
                                   error=str(e)[:500])
            failed_ids.append(sid)
            n_failed += 1
            tqdm.write(f"  FAILED {speaker['name_cleaned']}: {e}")

    # Quarantine failures into a failure batch
    if failed_ids:
        with get_conn() as conn:
            fb_id = create_failure_batch(
                conn, "query", failed_ids,
                name=f"query_failures_{run_id[:8]}",
                notes=f"Auto-created from run {run_id}",
            )
        print(f"\n  {n_failed} failures quarantined → failure_batch id={fb_id}")

    # Log run completion
    with get_conn() as conn:
        conn.execute(
            """UPDATE batch_runs
               SET finished_at=?, n_attempted=?, n_success=?, n_failed=?, n_skipped=?
               WHERE run_id=?""",
            (now_iso(), len(speakers), n_success, n_failed, n_skipped, run_id),
        )

    print(f"\nDone.  success={n_success}  failed={n_failed}  skipped={n_skipped}")


if __name__ == "__main__":
    main()
