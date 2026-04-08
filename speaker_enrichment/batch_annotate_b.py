"""
batch_annotate_b.py
===================
Submodule 2 — Group B annotation: political party affiliations.

Schema of annotation_json:
{
  "parties": [
    {
      "party_name":    "Labour Party",
      "party_abbrev":  "LAB",          // short name / acronym if known, else null
      "country":       "GB",           // ISO-2 if determinable
      "start_year":    1997,           // null if unknown
      "end_year":      null,           // null = current / unknown
      "notes":         "..."           // optional clarifying note
    },
    ...
  ],
  "n_parties":   1,
  "confidence": "high" | "medium" | "low"
}

Usage:
    python3 batch_annotate_b.py [--limit N] [--failure-batch-id ID]
"""

import argparse
import uuid

from tqdm import tqdm

import config
from db import (
    FAILED, SUCCESS, SKIPPED,
    activate_failure_batch, create_failure_batch,
    get_conn, init_db, now_iso, save_annotation, set_speaker_status,
)
from llm_client import (
    acquire_llm_lock, chat, extract_json,
    is_llm_locked, load_model, release_llm_lock,
)


GROUP = "B"

SYSTEM_PROMPT = """You are a data extraction assistant for a scientific study on politicians.

Given a biographical CV text, extract political party affiliations and return
a single JSON object.

Fields:
  "parties" : array of party objects, each with:
      "party_name"   : full party name (string)
      "party_abbrev" : abbreviation or short name (string or null)
      "country"      : ISO-2 country code of the party (string or null)
      "start_year"   : year joined the party (integer or null)
      "end_year"     : year left the party (integer or null; null = still member or unknown)
      "notes"        : any relevant clarifying note (string or null)
  "n_parties"  : total number of distinct party affiliations found (integer)
  "confidence" : "high", "medium", or "low"

If no party information is present, return {"parties": [], "n_parties": 0, "confidence": "high"}.
Respond ONLY with the JSON object."""


def annotate(cv_text: str, name: str) -> dict:
    user_msg = f"Person: {name}\n\nCV:\n{cv_text}\n\nExtract Group B (party) fields:"
    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_ANNOTATE_B,
        max_tokens=1024,
    )
    result = extract_json(response)
    result.setdefault("parties", [])
    result.setdefault("n_parties", len(result["parties"]))
    result.setdefault("confidence", None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=config.BATCH_SIZE_ANNOTATE)
    parser.add_argument("--failure-batch-id", type=int, default=None)
    args = parser.parse_args()

    if is_llm_locked():
        print("LLM is currently in use. Exiting.")
        return

    init_db()
    run_id = str(uuid.uuid4())

    try:
        acquire_llm_lock(f"annotate_{GROUP}", config.MODEL_ANNOTATE_B)
        print(f"Loading model {config.MODEL_ANNOTATE_B}...")
        load_model(config.MODEL_ANNOTATE_B)

        with get_conn() as conn:
            conn.execute(
                "INSERT INTO batch_runs (run_id, stage, batch_type, started_at) VALUES (?,?,?,?)",
                (run_id, f"annotate_{GROUP}",
                 "failure_retry" if args.failure_batch_id else "normal",
                 now_iso()),
            )

        with get_conn() as conn:
            if args.failure_batch_id:
                failure_ids = activate_failure_batch(conn, args.failure_batch_id)
                where_ids   = ",".join("?" * len(failure_ids))
                speakers    = conn.execute(
                    f"SELECT s.*, c.cv_text, c.created_at AS cv_created_at "
                    f"FROM speakers s JOIN speaker_cvs c USING(speaker_id) "
                    f"WHERE s.speaker_id IN ({where_ids})",
                    failure_ids,
                ).fetchall()
            else:
                speakers = conn.execute(
                    """SELECT s.*, c.cv_text, c.created_at AS cv_created_at
                       FROM speakers s
                       JOIN speaker_cvs c USING(speaker_id)
                       WHERE s.cv_synth_status = 'success'
                         AND s.annotate_b_status = 'pending'
                       LIMIT ?""",
                    (args.limit,),
                ).fetchall()

        print(f"Run {run_id[:8]}  |  {len(speakers)} speakers")

        n_success = n_failed = 0
        failed_ids: list[str] = []

        for speaker in tqdm(speakers, desc=f"Annotating group {GROUP}"):
            sid     = speaker["speaker_id"]
            name    = speaker["name_cleaned"] or "Unknown"
            cv_text = speaker["cv_text"] or ""

            with get_conn() as conn:
                set_speaker_status(conn, sid, "annotate_b", "running")

            if not cv_text.strip():
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, None, SKIPPED,
                                    config.MODEL_ANNOTATE_B,
                                    config.PROMPT_VERSION_ANNOTATE_B,
                                    cv_created_at=speaker["cv_created_at"],
                                    error="empty CV text")
                    set_speaker_status(conn, sid, "annotate_b", SKIPPED)
                continue

            try:
                result = annotate(cv_text, name)
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, result, SUCCESS,
                                    config.MODEL_ANNOTATE_B,
                                    config.PROMPT_VERSION_ANNOTATE_B,
                                    cv_created_at=speaker["cv_created_at"])
                    set_speaker_status(conn, sid, "annotate_b", SUCCESS,
                                       annotate_b_model=config.MODEL_ANNOTATE_B,
                                       annotate_b_prompt_v=config.PROMPT_VERSION_ANNOTATE_B)
                n_success += 1
            except Exception as e:
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, None, FAILED,
                                    config.MODEL_ANNOTATE_B,
                                    config.PROMPT_VERSION_ANNOTATE_B,
                                    error=str(e)[:500])
                    set_speaker_status(conn, sid, "annotate_b", FAILED, error=str(e)[:500])
                failed_ids.append(sid)
                n_failed += 1
                tqdm.write(f"  FAILED {name}: {e}")

        if failed_ids:
            with get_conn() as conn:
                fb_id = create_failure_batch(
                    conn, "annotate_b", failed_ids,
                    name=f"annotate_b_failures_{run_id[:8]}",
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
        release_llm_lock()


if __name__ == "__main__":
    main()
