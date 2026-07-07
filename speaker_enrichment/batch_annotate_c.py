"""
batch_annotate_c.py
===================
Submodule 2 — Group C annotation: education.

Scope: TERTIARY (higher-education) qualifications only, coded to ISCED 2011
levels. Non-tertiary education (primary, secondary, Matura/Abitur, vocational /
trade school, apprenticeships) is intentionally NOT recorded. Years, institution
country, and "elite" classification are also out of scope.

ISCED coding to levels makes education comparable across countries (e.g. an
Austrian Magister and a US master's both map to level 7), avoiding the earlier
catch-all "Other" bucket.

Schema of annotation_json:
{
  "education": [
    {
      "isced_level": 8 | 7 | 6 | 5 | null,     // 8 doctorate, 7 master/long-degree,
                                               // 6 bachelor, 5 short-cycle tertiary
      "field":       "Political Science",      // subject / discipline, or null
      "institution": "University of Cambridge" // or null
    },
    ...
  ],
  "highest_isced": 8 | 7 | 6 | 5 | null,       // computed in code = max of entries
  "confidence":    "high" | "medium" | "low"
}

Usage:
    python3 batch_annotate_c.py [--limit N] [--failure-batch-id ID]
"""

import argparse
import time
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


GROUP = "C"

SYSTEM_PROMPT = """You are a data extraction assistant for a scientific study on politicians.

From a biographical CV, extract ONLY the person's TERTIARY (higher / university-
level) education, and return a single JSON object. Code each tertiary
qualification with its ISCED 2011 level.

Record ONLY tertiary education. DO record university and other higher-education
qualifications. Do NOT record — ignore entirely — primary school, secondary
school, upper-secondary qualifications (Matura, Abitur, A-levels, baccalauréat,
high-school diploma), apprenticeships, and vocational / trade school. These are
not tertiary.

ISCED levels to assign:
  8 = Doctoral or equivalent (PhD, doctorate)
  7 = Master's or equivalent — includes Master's degrees AND the European long
      first degrees Magister and Diplom, and state-examination degrees in law or
      medicine
  6 = Bachelor's or equivalent
  5 = Short-cycle tertiary (e.g. associate degree, two-year technical /
      professional college, short Fachhochschule programmes, foundation degree)

Fields:
  "education" : array of tertiary entries, each with:
      "isced_level" : integer 5, 6, 7, or 8 (or null only if the qualification is
                      clearly tertiary but its level cannot be determined)
      "field"       : subject / discipline studied (string or null)
      "institution" : name of the university / higher-education institution (or null)
  "confidence"  : "high", "medium", or "low"

Rules:
- Only record education stated in the CV. Do NOT infer or invent qualifications,
  fields, or institutions.
- If the person has NO tertiary education stated, return
  {"education": [], "confidence": "high"}.
Respond ONLY with the JSON object.

(You do not need to compute the highest level — just list the tertiary qualifications.)"""


_VALID_ISCED = {5, 6, 7, 8}


def _coerce_isced(val) -> int | None:
    """Return the ISCED level as an int in {5,6,7,8}, else None."""
    try:
        lvl = int(val)
    except (ValueError, TypeError):
        return None
    return lvl if lvl in _VALID_ISCED else None


def annotate(cv_text: str, name: str) -> dict:
    user_msg = f"Person: {name}\n\nCV:\n{cv_text}\n\nExtract Group C (tertiary education) fields:"
    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_ANNOTATE_C,
        max_tokens=1024,
    )
    result = extract_json(response)

    # Keep only the fields in scope; coerce ISCED to a valid tertiary level.
    education = []
    for e in (result.get("education") or []):
        education.append({
            "isced_level": _coerce_isced(e.get("isced_level")),
            "field":       e.get("field"),
            "institution": e.get("institution"),
        })
    levels = [e["isced_level"] for e in education if e["isced_level"] is not None]
    return {
        "education":     education,
        "highest_isced": max(levels) if levels else None,   # computed, not model-reported
        "confidence":    result.get("confidence"),
    }


def _db_write_with_retry(fn, max_attempts: int = 10, base_delay: float = 1.0):
    """Run fn() (opens its own get_conn) with exponential backoff on DB lock."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise


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
        acquire_llm_lock(f"annotate_{GROUP}", config.MODEL_ANNOTATE_C)
        print(f"Loading model {config.MODEL_ANNOTATE_C}...")
        _loaded_instance = load_model(
            config.MODEL_ANNOTATE_C, context_length=config.LLM_CONTEXT_LENGTH
        ).get("instance_id")

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

        outcomes = {SUCCESS: 0, FAILED: 0, SKIPPED: 0}
        failed_ids: list[str] = []

        for speaker in tqdm(speakers, desc=f"Annotating group {GROUP}"):
            sid     = speaker["speaker_id"]
            name    = speaker["name_cleaned"] or "Unknown"
            cv_text = speaker["cv_text"] or ""

            with get_conn() as conn:
                set_speaker_status(conn, sid, "annotate_c", "running")

            if not cv_text.strip():
                def _skip(sid=sid, cva=speaker["cv_created_at"]):
                    with get_conn() as conn:
                        save_annotation(conn, sid, GROUP, None, SKIPPED,
                                        config.MODEL_ANNOTATE_C,
                                        config.PROMPT_VERSION_ANNOTATE_C,
                                        cv_created_at=cva, error="empty CV text")
                        set_speaker_status(conn, sid, "annotate_c", SKIPPED)
                _db_write_with_retry(_skip)
                outcomes[SKIPPED] += 1
                continue

            try:
                result = annotate(cv_text, name)
                def _save(sid=sid, result=result, cva=speaker["cv_created_at"]):
                    with get_conn() as conn:
                        save_annotation(conn, sid, GROUP, result, SUCCESS,
                                        config.MODEL_ANNOTATE_C,
                                        config.PROMPT_VERSION_ANNOTATE_C,
                                        cv_created_at=cva)
                        set_speaker_status(conn, sid, "annotate_c", SUCCESS,
                                           annotate_c_model=config.MODEL_ANNOTATE_C,
                                           annotate_c_prompt_v=config.PROMPT_VERSION_ANNOTATE_C)
                _db_write_with_retry(_save)
                outcomes[SUCCESS] += 1
            except Exception as e:
                err = str(e)[:500]
                def _fail(sid=sid, err=err, cva=speaker["cv_created_at"]):
                    with get_conn() as conn:
                        save_annotation(conn, sid, GROUP, None, FAILED,
                                        config.MODEL_ANNOTATE_C,
                                        config.PROMPT_VERSION_ANNOTATE_C,
                                        cv_created_at=cva, error=err)
                        set_speaker_status(conn, sid, "annotate_c", FAILED, error=err)
                try:
                    _db_write_with_retry(_fail)
                except Exception:
                    pass
                failed_ids.append(sid)
                outcomes[FAILED] += 1
                tqdm.write(f"  FAILED {name}: {e}")

        if failed_ids:
            with get_conn() as conn:
                fb_id = create_failure_batch(
                    conn, "annotate_c", failed_ids,
                    name=f"annotate_c_failures_{run_id[:8]}",
                )
            print(f"  {len(failed_ids)} failures → failure_batch id={fb_id}")

        with get_conn() as conn:
            conn.execute(
                """UPDATE batch_runs
                   SET finished_at=?, n_attempted=?, n_success=?, n_failed=?
                   WHERE run_id=?""",
                (now_iso(), len(speakers), outcomes[SUCCESS], outcomes[FAILED], run_id),
            )

        print(f"\nDone.  success={outcomes[SUCCESS]}  "
              f"skipped={outcomes[SKIPPED]}  failed={outcomes[FAILED]}")

    finally:
        if _loaded_instance:
            try:
                unload_model(_loaded_instance)
            except Exception:
                pass
        release_llm_lock()


if __name__ == "__main__":
    main()
