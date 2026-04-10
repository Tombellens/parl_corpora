"""
batch_annotate_a.py
===================
Submodule 2 — Group A annotation: gender, birth_year, birth_place.

Reads the merged CV text and extracts structured fields via local LLM.
Output JSON stored in speaker_annotations (group_name='A').

Schema of annotation_json:
{
  "gender":      "male" | "female" | "non-binary" | null,
  "birth_year":  1965 | null,
  "birth_place": "Brussels, Belgium" | null,
  "confidence":  "high" | "medium" | "low"
}

Usage:
    python3 batch_annotate_a.py [--limit N] [--failure-batch-id ID]
"""

import argparse
import json
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
    unload_model,
)


GROUP = "A"

SYSTEM_PROMPT = """You are a data extraction assistant for a scientific study on politicians.

Given a biographical CV text, extract the following information and return it
as a single JSON object.  Use null for any field not mentioned or unclear.

Fields to extract:
  "gender"      : "male", "female", "non-binary", or null
  "birth_year"  : integer year (e.g. 1965), or null
  "birth_place" : string (city and/or country), or null
  "confidence"  : your overall confidence: "high", "medium", or "low"

Respond ONLY with the JSON object, no explanation."""


def annotate(cv_text: str, name: str) -> dict:
    user_msg = f"Person: {name}\n\nCV:\n{cv_text}\n\nExtract Group A fields:"
    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_ANNOTATE_A,
        max_tokens=256,
    )
    result = extract_json(response)
    # Validate expected keys present
    for key in ("gender", "birth_year", "birth_place", "confidence"):
        result.setdefault(key, None)
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

    _loaded_instance = None
    try:
        acquire_llm_lock(f"annotate_{GROUP}", config.MODEL_ANNOTATE_A)
        print(f"Loading model {config.MODEL_ANNOTATE_A}...")
        _loaded_instance = load_model(config.MODEL_ANNOTATE_A).get("instance_id")
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
                         AND s.annotate_a_status = 'pending'
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
                set_speaker_status(conn, sid, f"annotate_{GROUP.lower()}", "running")

            if not cv_text.strip():
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, None, SKIPPED,
                                    config.MODEL_ANNOTATE_A,
                                    config.PROMPT_VERSION_ANNOTATE_A,
                                    cv_created_at=speaker["cv_created_at"],
                                    error="empty CV text")
                    set_speaker_status(conn, sid, "annotate_a", SKIPPED)
                continue

            try:
                result = annotate(cv_text, name)
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, result, SUCCESS,
                                    config.MODEL_ANNOTATE_A,
                                    config.PROMPT_VERSION_ANNOTATE_A,
                                    cv_created_at=speaker["cv_created_at"])
                    set_speaker_status(conn, sid, "annotate_a", SUCCESS,
                                       annotate_a_model=config.MODEL_ANNOTATE_A,
                                       annotate_a_prompt_v=config.PROMPT_VERSION_ANNOTATE_A)
                n_success += 1
            except Exception as e:
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, None, FAILED,
                                    config.MODEL_ANNOTATE_A,
                                    config.PROMPT_VERSION_ANNOTATE_A,
                                    error=str(e)[:500])
                    set_speaker_status(conn, sid, "annotate_a", FAILED, error=str(e)[:500])
                failed_ids.append(sid)
                n_failed += 1
                tqdm.write(f"  FAILED {name}: {e}")

        if failed_ids:
            with get_conn() as conn:
                fb_id = create_failure_batch(
                    conn, "annotate_a", failed_ids,
                    name=f"annotate_a_failures_{run_id[:8]}",
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
