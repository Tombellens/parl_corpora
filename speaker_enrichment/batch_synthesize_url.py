"""
batch_synthesize_url.py
=======================
Submodule 1, Part 3 — Per-URL LLM synthesis.

For each speaker whose pages are fetched, run the local LLM over each
successfully-fetched URL's cleaned text to produce a focused biographical
snippet.  The snippet retains everything potentially relevant:
  gender, birth year/place, party affiliations (with dates), education
  (degree, subject, institution), career steps (role, org, years).

Acquires the LLM lock before starting; releases it on exit.
All raw cleaned_text and synthesis_text are stored in speaker_urls.

Usage:
    python3 batch_synthesize_url.py [--limit N] [--failure-batch-id ID]
"""

import argparse
import uuid

from tqdm import tqdm

import config
from db import (
    FAILED, SUCCESS, SKIPPED,
    activate_failure_batch, create_failure_batch,
    fetch_pending_speakers, get_conn, init_db,
    now_iso, set_speaker_status,
)
from llm_client import (
    acquire_llm_lock, chat, extract_json,
    is_llm_locked, load_model, release_llm_lock,
    unload_model,
)


SYSTEM_PROMPT = """You are a research assistant extracting biographical information
about politicians from web pages for a scientific study.

STEP 1 — RELEVANCE CHECK
Before doing anything else, decide: is this page primarily about the person
named in the "Person:" field? A page is relevant if it is clearly a biography,
profile, or article whose main subject is that specific individual.

A page is NOT relevant if:
- It is primarily about a different person who shares the same name
- The person is only mentioned in passing (e.g. in a list, a quote, a footnote)
- It is a search results page, directory listing, or disambiguation page
- It is about an organisation, event, or topic and the person is incidental

If the page is NOT relevant, respond with exactly one word: IRRELEVANT

STEP 2 — BIOGRAPHICAL SUMMARY (only if relevant)
Write a concise factual summary containing ALL of the following elements
IF present in the text:

- Full name and any alternative spellings
- Gender
- Date of birth (year at minimum) and place of birth
- Political party or parties (with approximate joining/leaving years if mentioned)
- Educational background: degree(s) obtained, field(s) of study, institution(s)
- Career history: each position held, the organisation, and approximate years
- Any other politically relevant biographical facts

Write in plain prose, one paragraph. Do NOT add information not in the source.
If a piece of information is absent from the text, simply omit it.
Do NOT output JSON. Output only the prose summary."""


def synthesize_url(url_row: dict, speaker_name: str) -> str:
    """
    Call the LLM to synthesize one URL's cleaned text.
    Returns empty string if the LLM judges the page irrelevant to the speaker.
    """
    cleaned = url_row["cleaned_text"] or ""
    if not cleaned.strip():
        return ""

    user_msg = (
        f"Person: {speaker_name}\n"
        f"Source URL: {url_row['url']}\n\n"
        f"--- PAGE TEXT ---\n{cleaned}\n--- END ---\n\n"
        "First check relevance, then write the biographical summary if relevant:"
    )

    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_SYNTHESIZE_URL,
        max_tokens=1024,
    )

    # Treat IRRELEVANT responses as empty (will be marked SKIPPED)
    if response.strip().upper().startswith("IRRELEVANT"):
        return ""
    return response


def process_speaker(conn, speaker: dict) -> tuple[int, int]:
    """
    Synthesize all pending URL rows for this speaker.
    Returns (n_done, n_failed).
    """
    sid  = speaker["speaker_id"]
    name = speaker["name_cleaned"] or "Unknown"

    pending_urls = conn.execute(
        """SELECT * FROM speaker_urls
           WHERE speaker_id = ? AND fetch_status = 'success'
             AND synthesis_status = 'pending'""",
        (sid,),
    ).fetchall()

    if not pending_urls:
        set_speaker_status(conn, sid, "url_synth", SKIPPED,
                           error="no successfully fetched URLs")
        return 0, 0

    n_done = n_failed = 0
    for url_row in pending_urls:
        try:
            snippet = synthesize_url(url_row, name)
            if not snippet.strip():
                conn.execute(
                    "UPDATE speaker_urls SET synthesis_status=?, synthesis_at=? WHERE id=?",
                    (SKIPPED, now_iso(), url_row["id"]),
                )
            else:
                conn.execute(
                    """UPDATE speaker_urls
                       SET synthesis_status=?, synthesis_at=?, synthesis_model=?,
                           synthesis_prompt_v=?, synthesis_text=?
                       WHERE id=?""",
                    (SUCCESS, now_iso(),
                     config.MODEL_SYNTHESIZE_URL,
                     config.PROMPT_VERSION_SYNTHESIZE_URL,
                     snippet, url_row["id"]),
                )
            n_done += 1
        except Exception as e:
            conn.execute(
                "UPDATE speaker_urls SET synthesis_status=?, synthesis_error=? WHERE id=?",
                (FAILED, str(e)[:500], url_row["id"]),
            )
            n_failed += 1

    # Update speaker-level status
    any_synth = conn.execute(
        "SELECT COUNT(*) FROM speaker_urls WHERE speaker_id=? AND synthesis_status='success'",
        (sid,),
    ).fetchone()[0]

    if any_synth:
        set_speaker_status(conn, sid, "url_synth", SUCCESS,
                           url_synth_model=config.MODEL_SYNTHESIZE_URL,
                           url_synth_prompt_v=config.PROMPT_VERSION_SYNTHESIZE_URL,
                           url_synth_n_done=any_synth)
    elif n_failed > 0:
        set_speaker_status(conn, sid, "url_synth", FAILED,
                           error=f"{n_failed} URL(s) failed synthesis")
    else:
        set_speaker_status(conn, sid, "url_synth", SKIPPED,
                           error="no fetchable URLs yielded text")

    return n_done, n_failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=config.BATCH_SIZE_SYNTHESIZE_URL)
    parser.add_argument("--failure-batch-id", type=int, default=None)
    args = parser.parse_args()

    if is_llm_locked():
        print("LLM is currently in use by another process. Exiting.")
        return

    init_db()
    run_id = str(uuid.uuid4())

    _loaded_instance = None
    try:
        acquire_llm_lock("synthesize_url", config.MODEL_SYNTHESIZE_URL)
        print(f"Loading model {config.MODEL_SYNTHESIZE_URL}...")
        _loaded_instance = load_model(config.MODEL_SYNTHESIZE_URL).get("instance_id")
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO batch_runs (run_id, stage, batch_type, started_at) VALUES (?,?,?,?)",
                (run_id, "url_synth",
                 "failure_retry" if args.failure_batch_id else "normal",
                 now_iso()),
            )

        with get_conn() as conn:
            if args.failure_batch_id:
                failure_ids = activate_failure_batch(conn, args.failure_batch_id)
                speakers    = fetch_pending_speakers(conn, "url_synth", args.limit,
                                                     failure_ids=failure_ids)
            else:
                speakers = conn.execute(
                    """SELECT * FROM speakers
                       WHERE fetch_status = 'success' AND url_synth_status = 'pending'
                       LIMIT ?""",
                    (args.limit,),
                ).fetchall()

        print(f"Run {run_id[:8]}  |  {len(speakers)} speakers")

        n_success = n_failed = 0
        failed_ids: list[str] = []

        for speaker in tqdm(speakers, desc="Synthesizing URLs"):
            sid = speaker["speaker_id"]
            with get_conn() as conn:
                set_speaker_status(conn, sid, "url_synth", "running")
            try:
                with get_conn() as conn:
                    process_speaker(conn, speaker)
                n_success += 1
            except Exception as e:
                with get_conn() as conn:
                    set_speaker_status(conn, sid, "url_synth", FAILED, error=str(e)[:500])
                failed_ids.append(sid)
                n_failed += 1
                tqdm.write(f"  FAILED {speaker['name_cleaned']}: {e}")

        if failed_ids:
            with get_conn() as conn:
                fb_id = create_failure_batch(
                    conn, "url_synth", failed_ids,
                    name=f"url_synth_failures_{run_id[:8]}",
                )
            print(f"  {n_failed} failures → failure_batch id={fb_id}")

        with get_conn() as conn:
            conn.execute(
                """UPDATE batch_runs
                   SET finished_at=?, n_attempted=?, n_success=?, n_failed=?
                   WHERE run_id=?""",
                (now_iso(), len(speakers), n_success, n_failed, run_id),
            )

        print(f"\nDone.  success={n_success}  failed={n_failed}")

    finally:
        if _loaded_instance:
            try:
                unload_model(_loaded_instance)
            except Exception:
                pass
        release_llm_lock()


if __name__ == "__main__":
    main()
