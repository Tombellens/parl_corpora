"""
batch_annotate_c.py
===================
Submodule 2 — Group C annotation: education.

Scope (deliberately lean): highest degree, field(s) of study, institution(s).
Years, institution country, and "elite" classification are intentionally NOT
coded here — elite-ness, if needed, is a researcher-defined classification best
applied deterministically against a reference list in a later step, not judged
inline by the model.

Schema of annotation_json:
{
  "education": [
    {
      "degree":      "PhD" | "Master" | "Bachelor" | "Other" | null,
      "field":       "Political Science",      // subject / discipline, or null
      "institution": "University of Cambridge" // or null
    },
    ...
  ],
  "highest_degree": "PhD" | "Master" | "Bachelor" | "Other" | null,
  "confidence":     "high" | "medium" | "low"
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

Given a biographical CV text, extract the person's educational background and
return a single JSON object.

Fields:
  "education" : array of education entries, one per qualification, each with:
      "degree"      : the qualification level — one of "PhD", "Master",
                      "Bachelor", "Other", or null
      "field"       : subject / discipline studied (string or null)
      "institution" : name of the university, school, or training body (or null)
  "confidence"     : "high", "medium", or "low"

What to record:
- University degrees: doctorate → "PhD"; master's / Magister / Diplom → "Master";
  bachelor's → "Bachelor".
- ALSO record non-university education as degree "Other": apprenticeships,
  vocational / trade school (e.g. Handelsschule, Fachschule), professional
  training, academies, and completed secondary qualifications (e.g. Matura /
  Abitur) when they are the person's notable education. Use "Other" for these.
- Return an entry per distinct qualification. An entry with degree "Other" is
  still a valid entry.

Rules:
- Only record education that is stated in the CV. Do NOT infer or invent
  degrees, fields, or institutions.
- Use the institution / training-body name as given in the CV.
- If NO education of any kind is stated, return
  {"education": [], "confidence": "high"}.
Respond ONLY with the JSON object.

(You do not need to compute a highest degree — just list the qualifications.)"""


# Degree ranking for the deterministic highest_degree rollup.
_DEGREE_RANK = {"PhD": 4, "Master": 3, "Bachelor": 2, "Other": 1}


def _highest_degree(education: list[dict]) -> str | None:
    """Top qualification across entries: PhD > Master > Bachelor > Other.
    Computed in code so the rollup is always consistent with the entries
    (the model is not trusted to rank them)."""
    best = None
    best_rank = 0
    for e in education:
        d = e.get("degree")
        r = _DEGREE_RANK.get(d, 0)
        if r > best_rank:
            best_rank, best = r, d
    return best


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

    # Keep only the fields in scope; drop anything extra the model may add.
    education = []
    for e in (result.get("education") or []):
        education.append({
            "degree":      e.get("degree"),
            "field":       e.get("field"),
            "institution": e.get("institution"),
        })
    return {
        "education":      education,
        "highest_degree": _highest_degree(education),   # computed, not model-reported
        "confidence":     result.get("confidence"),
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
