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
import time
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
about parliamentarians (members of parliament / congress / national legislatures)
from web pages for a scientific study.

The "Person:" field names a specific politician. Your job has two steps.

STEP 1 — RELEVANCE CHECK (do this first, every time)
Decide whether this page is primarily about THAT specific politician.

The page IS relevant only if its main subject is that individual politician
(a biography, official profile, parliamentary record, candidate page, or an
article clearly about them).

The page is NOT relevant if ANY of the following is true:
- It is mainly about a DIFFERENT person who happens to share the same name
  (e.g. an athlete, businessperson, actor, academic, or private individual who
  is not this parliamentarian)
- The named person appears only in passing — in a list, a quote, a caption, a
  comment thread, or a footnote — and is not the focus
- It is a search-results page, index, tag page, disambiguation page, or
  navigation/boilerplate
- It is about an organisation, event, law, or topic and the person is incidental
- You cannot find enough on the page to confirm it is about this politician

If the page is NOT relevant, respond with EXACTLY this one word and nothing else:
IRRELEVANT

Do not guess or fabricate to make a page seem relevant. When in doubt, answer
IRRELEVANT.

STEP 2 — BIOGRAPHICAL SUMMARY (only if the page passed STEP 1)
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


# ---------------------------------------------------------------------------
# Context-length budgeting
# ---------------------------------------------------------------------------
# We want to send the ENTIRE page text whenever it fits the model context.
# Only when a text is so long it would overflow the context do we truncate it
# to fit (and flag that we did so), rather than letting the LLM 400 and lose
# the whole call. Token count is estimated from UTF-8 byte length, which is far
# more stable across scripts (Latin/Cyrillic/Greek/CJK) than character count.
_OUTPUT_TOKENS_RESERVED = 1536    # room for the model's answer
_PROMPT_OVERHEAD_TOKENS = 1200    # system prompt + user wrapper, generous
_BYTES_PER_TOKEN        = 3.0     # conservative (real BPE ≈ 3.5–4 bytes/token)


def _max_input_bytes() -> int:
    """Max UTF-8 bytes of page text we can send without risking overflow."""
    input_token_budget = (
        config.LLM_CONTEXT_LENGTH
        - _OUTPUT_TOKENS_RESERVED
        - _PROMPT_OVERHEAD_TOKENS
    )
    return max(0, int(input_token_budget * _BYTES_PER_TOKEN))


def _fit_to_context(text: str) -> tuple[str, bool]:
    """
    Return (text, was_truncated). If the text fits the context budget it is
    returned unchanged. Otherwise it is truncated on a UTF-8 boundary to fit,
    and was_truncated=True so the caller can flag it for review.
    """
    budget = _max_input_bytes()
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text, False
    truncated = encoded[:budget].decode("utf-8", errors="ignore")
    return truncated + "\n[TRUNCATED TO FIT MODEL CONTEXT]", True


def synthesize_url(url_row: dict, speaker_name: str) -> tuple[str, str | None]:
    """
    Call the LLM to synthesize one URL's cleaned text.

    Returns (snippet, note):
      - snippet is "" if the page text is empty or the LLM judges the page
        IRRELEVANT to the speaker.
      - note is None normally, or an informational string (e.g. a truncation
        flag) stored alongside the result for scientific traceability.
    """
    cleaned = url_row["cleaned_text"] or ""
    if not cleaned.strip():
        return "", None

    text, was_truncated = _fit_to_context(cleaned)
    note = None
    if was_truncated:
        note = (f"input_truncated_to_context: original {len(cleaned.encode('utf-8'))} "
                f"bytes exceeded budget {_max_input_bytes()} bytes")

    user_msg = (
        f"Person: {speaker_name}\n"
        f"Source URL: {url_row['url']}\n\n"
        f"--- PAGE TEXT ---\n{text}\n--- END ---\n\n"
        "First check relevance, then write the biographical summary if relevant:"
    )

    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_SYNTHESIZE_URL,
        max_tokens=_OUTPUT_TOKENS_RESERVED,
    )

    # Treat IRRELEVANT responses as empty (will be marked SKIPPED)
    if response.strip().upper().startswith("IRRELEVANT"):
        return "", note
    return response, note


def _db_write_with_retry(fn, max_attempts: int = 10, base_delay: float = 1.0):
    """Run fn() (opens its own get_conn) with exponential backoff on DB lock."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                tqdm.write(f"  DB locked, retrying in {delay:.1f}s "
                           f"(attempt {attempt+1}/{max_attempts})")
                time.sleep(delay)
            else:
                raise


def process_speaker(speaker: dict) -> tuple[int, int]:
    """
    Synthesize all pending URL rows for this speaker.

    Each URL's synthesis result is committed independently (its own
    transaction, with lock retry), so an LLM error on one URL — or a DB lock —
    can never discard the GPU-expensive synthesis already done for the other
    URLs of this speaker. The speaker-level status is computed from committed
    rows in its own transaction.

    Returns (n_done, n_failed).
    """
    sid  = speaker["speaker_id"]
    name = speaker["name_cleaned"] or "Unknown"

    with get_conn() as conn:
        pending_urls = conn.execute(
            """SELECT * FROM speaker_urls
               WHERE speaker_id = ? AND fetch_status = 'success'
                 AND synthesis_status = 'pending'""",
            (sid,),
        ).fetchall()

    n_done = n_failed = 0
    for url_row in pending_urls:
        uid = url_row["id"]
        try:
            snippet, note = synthesize_url(url_row, name)
            if not snippet.strip():
                def _save_skip(uid=uid, note=note):
                    with get_conn() as conn:
                        conn.execute(
                            """UPDATE speaker_urls
                               SET synthesis_status=?, synthesis_at=?, synthesis_error=?
                               WHERE id=?""",
                            (SKIPPED, now_iso(), note, uid),
                        )
                _db_write_with_retry(_save_skip)
            else:
                def _save_ok(uid=uid, snippet=snippet, note=note):
                    with get_conn() as conn:
                        conn.execute(
                            """UPDATE speaker_urls
                               SET synthesis_status=?, synthesis_at=?, synthesis_model=?,
                                   synthesis_prompt_v=?, synthesis_text=?, synthesis_error=?
                               WHERE id=?""",
                            (SUCCESS, now_iso(),
                             config.MODEL_SYNTHESIZE_URL,
                             config.PROMPT_VERSION_SYNTHESIZE_URL,
                             snippet, note, uid),
                        )
                _db_write_with_retry(_save_ok)
            n_done += 1
        except Exception as e:
            err = str(e)[:500]
            def _save_fail(uid=uid, err=err):
                with get_conn() as conn:
                    conn.execute(
                        "UPDATE speaker_urls SET synthesis_status=?, synthesis_error=? WHERE id=?",
                        (FAILED, err, uid),
                    )
            _db_write_with_retry(_save_fail)
            n_failed += 1
            tqdm.write(f"  URL synth failed ({name}): {err}")

    # Speaker-level aggregate, from committed URL rows, in its own transaction.
    def _set_aggregate():
        with get_conn() as conn:
            any_synth = conn.execute(
                "SELECT COUNT(*) FROM speaker_urls WHERE speaker_id=? AND synthesis_status='success'",
                (sid,),
            ).fetchone()[0]
            still_pending = conn.execute(
                "SELECT COUNT(*) FROM speaker_urls WHERE speaker_id=? AND fetch_status='success' AND synthesis_status='pending'",
                (sid,),
            ).fetchone()[0]

            if any_synth:
                set_speaker_status(conn, sid, "url_synth", SUCCESS,
                                   url_synth_model=config.MODEL_SYNTHESIZE_URL,
                                   url_synth_prompt_v=config.PROMPT_VERSION_SYNTHESIZE_URL,
                                   url_synth_n_done=any_synth)
            elif still_pending > 0:
                # Some URLs not yet synthesized (e.g. failed mid-run) — keep
                # pending so a rerun retries, never falsely mark done.
                set_speaker_status(conn, sid, "url_synth", "pending")
            elif n_failed > 0:
                set_speaker_status(conn, sid, "url_synth", FAILED,
                                   error=f"{n_failed} URL(s) failed synthesis")
            else:
                set_speaker_status(conn, sid, "url_synth", SKIPPED,
                                   error="no fetchable URLs yielded text")
    _db_write_with_retry(_set_aggregate)

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
        _loaded_instance = load_model(config.MODEL_SYNTHESIZE_URL, context_length=config.LLM_CONTEXT_LENGTH).get("instance_id")
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
                # process_speaker commits each URL synthesis and the aggregate
                # status independently, so no GPU work is lost if this raises.
                process_speaker(speaker)
                n_success += 1
            except Exception as e:
                # URL-level results are already committed; reset the speaker to
                # pending so a rerun re-finalises it rather than silently failing.
                try:
                    with get_conn() as conn:
                        set_speaker_status(conn, sid, "url_synth", "pending", error=str(e)[:500])
                except Exception:
                    pass
                failed_ids.append(sid)
                n_failed += 1
                tqdm.write(f"  ERROR finalising {speaker['name_cleaned']}: {e}")

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

        # Honest summary: report ACTUAL synthesis outcomes from the DB for the
        # speakers in this run, not just "speaker processed without exception".
        speaker_ids = [s["speaker_id"] for s in speakers]
        if speaker_ids:
            with get_conn() as conn:
                ph = ",".join("?" * len(speaker_ids))
                url_rows = conn.execute(
                    f"""SELECT synthesis_status, COUNT(*)
                        FROM speaker_urls
                        WHERE speaker_id IN ({ph})
                        GROUP BY synthesis_status""",
                    speaker_ids,
                ).fetchall()
                spk_rows = conn.execute(
                    f"""SELECT url_synth_status, COUNT(*)
                        FROM speakers WHERE speaker_id IN ({ph})
                        GROUP BY url_synth_status""",
                    speaker_ids,
                ).fetchall()
            url_counts = {r[0]: r[1] for r in url_rows}
            spk_counts = {r[0]: r[1] for r in spk_rows}
            print(f"\nDone.  speakers processed={n_success}  finalise-errors={n_failed}")
            print(f"  URL synthesis: " + "  ".join(f"{k}={v}" for k, v in sorted(url_counts.items())))
            print(f"  Speaker status: " + "  ".join(f"{k}={v}" for k, v in sorted(spk_counts.items())))
        else:
            print(f"\nDone.  nothing to process.")

    finally:
        if _loaded_instance:
            try:
                unload_model(_loaded_instance)
            except Exception:
                pass
        release_llm_lock()


if __name__ == "__main__":
    main()
