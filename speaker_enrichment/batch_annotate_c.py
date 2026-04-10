"""
batch_annotate_c.py
===================
Submodule 2 — Group C annotation: education.

Schema of annotation_json:
{
  "education": [
    {
      "degree":          "PhD" | "Master" | "Bachelor" | "Other" | null,
      "field":           "Political Science",   // subject / discipline
      "institution":     "University of Cambridge",
      "institution_country": "GB",              // ISO-2 or null
      "year_start":      1985,                  // null if unknown
      "year_end":        1988,                  // null if unknown
      "elite_institution": true | false | null  // Oxbridge, Ivy League, Grandes Écoles, etc.
    },
    ...
  ],
  "highest_degree":    "PhD" | "Master" | "Bachelor" | "Other" | null,
  "any_elite_institution": true | false | null,
  "confidence": "high" | "medium" | "low"
}

Usage:
    python3 batch_annotate_c.py [--limit N] [--failure-batch-id ID]
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
    unload_all_models,
)


GROUP = "C"

# Elite institutions list (non-exhaustive, for prompt guidance)
ELITE_EXAMPLES = (
    "Oxford, Cambridge, Harvard, Yale, Princeton, MIT, Columbia, Stanford, "
    "LSE, Sciences Po, ENA/INSP, École Polytechnique, HEC Paris, "
    "Bocconi, LMU Munich, ETH Zurich, Leiden, KU Leuven"
)

SYSTEM_PROMPT = f"""You are a data extraction assistant for a scientific study on politicians.

Given a biographical CV text, extract educational background and return a single JSON object.

Fields:
  "education" : array of education entries, each with:
      "degree"              : highest degree at that institution — one of:
                              "PhD", "Master", "Bachelor", "Other", or null
      "field"               : subject / discipline studied (string or null)
      "institution"         : name of university or school (string or null)
      "institution_country" : ISO-2 country code (string or null)
      "year_start"          : year started (integer or null)
      "year_end"            : year finished / graduated (integer or null)
      "elite_institution"   : true if the institution is considered elite / prestigious
                              (examples: {ELITE_EXAMPLES}), false otherwise, null if unknown
  "highest_degree"        : the highest degree across all entries
                            ("PhD", "Master", "Bachelor", "Other", or null)
  "any_elite_institution" : true if any entry has elite_institution=true, else false, null if unknown
  "confidence"            : "high", "medium", or "low"

If no education information is present, return:
  {{"education": [], "highest_degree": null, "any_elite_institution": null, "confidence": "high"}}
Respond ONLY with the JSON object."""


def annotate(cv_text: str, name: str) -> dict:
    user_msg = f"Person: {name}\n\nCV:\n{cv_text}\n\nExtract Group C (education) fields:"
    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_ANNOTATE_C,
        max_tokens=1024,
    )
    result = extract_json(response)
    result.setdefault("education", [])
    result.setdefault("highest_degree", None)
    result.setdefault("any_elite_institution", None)
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
        acquire_llm_lock(f"annotate_{GROUP}", config.MODEL_ANNOTATE_C)
        print(f"Loading model {config.MODEL_ANNOTATE_C}...")
        load_model(config.MODEL_ANNOTATE_C)

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
                         AND s.annotate_c_status = 'pending'
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
                set_speaker_status(conn, sid, "annotate_c", "running")

            if not cv_text.strip():
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, None, SKIPPED,
                                    config.MODEL_ANNOTATE_C,
                                    config.PROMPT_VERSION_ANNOTATE_C,
                                    cv_created_at=speaker["cv_created_at"],
                                    error="empty CV text")
                    set_speaker_status(conn, sid, "annotate_c", SKIPPED)
                continue

            try:
                result = annotate(cv_text, name)
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, result, SUCCESS,
                                    config.MODEL_ANNOTATE_C,
                                    config.PROMPT_VERSION_ANNOTATE_C,
                                    cv_created_at=speaker["cv_created_at"])
                    set_speaker_status(conn, sid, "annotate_c", SUCCESS,
                                       annotate_c_model=config.MODEL_ANNOTATE_C,
                                       annotate_c_prompt_v=config.PROMPT_VERSION_ANNOTATE_C)
                n_success += 1
            except Exception as e:
                with get_conn() as conn:
                    save_annotation(conn, sid, GROUP, None, FAILED,
                                    config.MODEL_ANNOTATE_C,
                                    config.PROMPT_VERSION_ANNOTATE_C,
                                    error=str(e)[:500])
                    set_speaker_status(conn, sid, "annotate_c", FAILED, error=str(e)[:500])
                failed_ids.append(sid)
                n_failed += 1
                tqdm.write(f"  FAILED {name}: {e}")

        if failed_ids:
            with get_conn() as conn:
                fb_id = create_failure_batch(
                    conn, "annotate_c", failed_ids,
                    name=f"annotate_c_failures_{run_id[:8]}",
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
        unload_all_models()
        release_llm_lock()


if __name__ == "__main__":
    main()
