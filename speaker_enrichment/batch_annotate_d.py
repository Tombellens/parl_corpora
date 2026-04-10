"""
batch_annotate_d.py
===================
Submodule 2 — Group D annotation: career steps.

Schema of annotation_json:
{
  "career": [
    {
      "job_title":     "Minister of Finance",
      "organisation":  "Government of France",
      "sector":        "government" | "legislature" | "judiciary" | "military" |
                       "business" | "academia" | "ngo" | "media" | "party" |
                       "other" | null,
      "start_year":    2012,
      "end_year":      2017,         // null = current or unknown
      "notes":         null
    },
    ...
  ],
  "n_positions":  5,
  "confidence":  "high" | "medium" | "low"
}

Sector codes (use exactly these strings):
  government  — executive branch (minister, secretary, mayor, …)
  legislature — parliament, senate, assembly member, …
  judiciary   — judge, magistrate, …
  military    — army, navy, air force officer, …
  business    — private sector, corporate, entrepreneur
  academia    — professor, researcher, university
  ngo         — civil society, think tank, union, association
  media       — journalist, broadcaster, editor
  party       — party official / secretary (not as elected MP)
  other       — anything else

Usage:
    python3 batch_annotate_d.py [--limit N] [--failure-batch-id ID]
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
    unload_model,
)


GROUP = "D"

SYSTEM_PROMPT = """You are a data extraction assistant for a scientific study on politicians.

Given a biographical CV text, extract the career history and return a single JSON object.

Fields:
  "career" : array of career step objects (chronological order, earliest first), each with:
      "job_title"    : position or role title (string or null)
      "organisation" : employer / institution name (string or null)
      "sector"       : one of the following codes (string or null):
                       "government", "legislature", "judiciary", "military",
                       "business", "academia", "ngo", "media", "party", "other"
      "start_year"   : year started (integer or null)
      "end_year"     : year ended (integer or null; null = current or unknown)
      "notes"        : brief clarifying note if needed (string or null)
  "n_positions" : total number of career steps found (integer)
  "confidence"  : "high", "medium", or "low"

If no career information is present, return:
  {"career": [], "n_positions": 0, "confidence": "high"}

Important:
- Parliamentary mandates (MP, senator, MEP) → sector "legislature"
- Ministerial roles → sector "government"
- Party leadership roles without electoral mandate → sector "party"
- List steps chronologically; include overlapping roles as separate entries

Respond ONLY with the JSON object."""


def annotate(cv_text: str, name: str) -> dict:
    user_msg = f"Person: {name}\n\nCV:\n{cv_text}\n\nExtract Group D (career) fields:"
    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_ANNOTATE_D,
        max_tokens=2048,
    )
    result = extract_json(response)
    result.setdefault("career", [])
    result.setdefault("n_positions", len(result["career"]))
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

    _loaded_instance = None
    try:
        acquire_llm_lock(f"annotate_{GROUP}", config.MODEL_ANNOTATE_D)
        print(f"Loading model {config.MODEL_ANNOTATE_D}...")
        _loaded_instance = load_model(config.MODEL_ANNOTATE_D).get("instance_id")

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
                         AND s.annotate_d_status = 'pending'
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
                set_speaker_status(conn, sid, "annotate_d", "running")

            if not cv_text.strip():
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, None, SKIPPED,
                                    config.MODEL_ANNOTATE_D,
                                    config.PROMPT_VERSION_ANNOTATE_D,
                                    cv_created_at=speaker["cv_created_at"],
                                    error="empty CV text")
                    set_speaker_status(conn, sid, "annotate_d", SKIPPED)
                continue

            try:
                result = annotate(cv_text, name)
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, result, SUCCESS,
                                    config.MODEL_ANNOTATE_D,
                                    config.PROMPT_VERSION_ANNOTATE_D,
                                    cv_created_at=speaker["cv_created_at"])
                    set_speaker_status(conn, sid, "annotate_d", SUCCESS,
                                       annotate_d_model=config.MODEL_ANNOTATE_D,
                                       annotate_d_prompt_v=config.PROMPT_VERSION_ANNOTATE_D)
                n_success += 1
            except Exception as e:
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, None, FAILED,
                                    config.MODEL_ANNOTATE_D,
                                    config.PROMPT_VERSION_ANNOTATE_D,
                                    error=str(e)[:500])
                    set_speaker_status(conn, sid, "annotate_d", FAILED, error=str(e)[:500])
                failed_ids.append(sid)
                n_failed += 1
                tqdm.write(f"  FAILED {name}: {e}")

        if failed_ids:
            with get_conn() as conn:
                fb_id = create_failure_batch(
                    conn, "annotate_d", failed_ids,
                    name=f"annotate_d_failures_{run_id[:8]}",
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
