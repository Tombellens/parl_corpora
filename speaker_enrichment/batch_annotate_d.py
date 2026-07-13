"""
batch_annotate_d.py
===================
Submodule 2 — Group D annotation: sectors of professional experience.

Simplified design: instead of a full career timeline (titles, organisations,
dates), we record only the SET of high-level sectors in which the person had
professional / occupational experience before or during their time as an MP.
No durations, no titles — just a deduplicated set of sector codes from a fixed
codebook. The national parliamentary mandate itself is NOT coded (it is constant
across all speakers).

Schema of annotation_json:
{
  "sectors": ["business", "politics_party", ...],   // deduplicated, from the codebook
  "confidence": "high" | "medium" | "low"
}

Usage:
    python3 batch_annotate_d.py [--limit N] [--failure-batch-id ID]
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


GROUP = "D"

# Fixed high-level sector codebook (order = canonical output order).
SECTOR_ORDER = [
    "public_administration",
    "politics_party",
    "law",
    "business",
    "academia_education",
    "media",
    "civil_society",
    "military_security",
    "other",
]
SECTOR_CODES = set(SECTOR_ORDER)

SYSTEM_PROMPT = """You are a data extraction assistant for a scientific study on politicians.

From a biographical CV, identify the SECTORS in which the person had professional
or occupational experience before or during their time as a member of parliament.
Return a single JSON object with a deduplicated set of sector codes. Do NOT report
job titles, organisations, durations, or dates — only the set of sectors.

Use ONLY these sector codes:
  "public_administration" : civil service, public-sector agencies, non-elected
                            government / administrative roles, diplomacy
  "politics_party"        : professional party or political roles (functionary,
                            adviser, political staff) and prior elected office at
                            local, regional, or European level
  "law"                   : legal profession — lawyer, judge, prosecutor, notary
  "business"              : private sector — employee, manager, executive,
                            entrepreneur, self-employed, farming, skilled trades
  "academia_education"    : universities, research, teaching at any level
  "media"                 : journalism, broadcasting, publishing, communications / PR
  "civil_society"         : NGOs, trade unions, interest / advocacy groups,
                            foundations, charities, religious organisations
  "military_security"     : armed forces, police, intelligence, security services
  "other"                 : any occupation not covered above (e.g. healthcare, arts, sports)

Rules:
- Include a sector if the CV shows the person worked in it at any point before or
  during their parliamentary career.
- Do NOT code the national parliamentary mandate itself (being an MP / senator /
  member of the national parliament). Code only other sectors of experience.
- Base sectors only on what the CV states. Do NOT infer experience with no basis.
- Return each sector at most once (a set, not a list of positions).
- If the CV shows no professional experience outside the parliamentary mandate,
  return {"sectors": [], "confidence": "high"}.

Output JSON:
{
  "sectors": ["business", "politics_party", ...],
  "confidence": "high" | "medium" | "low"
}
Respond ONLY with the JSON object."""


def annotate(cv_text: str, name: str) -> dict:
    user_msg = f"Person: {name}\n\nCV:\n{cv_text}\n\nList Group D (sectors of experience):"
    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_ANNOTATE_D,
        max_tokens=256,
    )
    result = extract_json(response)

    # Validate against the codebook, deduplicate, and order canonically.
    found = set()
    for s in (result.get("sectors") or []):
        if isinstance(s, str):
            code = s.strip().lower()
            if code in SECTOR_CODES:
                found.add(code)
    sectors = [c for c in SECTOR_ORDER if c in found]

    return {"sectors": sectors, "confidence": result.get("confidence")}


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
        acquire_llm_lock(f"annotate_{GROUP}", config.MODEL_ANNOTATE_D)
        print(f"Loading model {config.MODEL_ANNOTATE_D}...")
        _loaded_instance = load_model(
            config.MODEL_ANNOTATE_D, context_length=config.LLM_CONTEXT_LENGTH
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
                         AND s.annotate_d_status = 'pending'
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
                set_speaker_status(conn, sid, "annotate_d", "running")

            if not cv_text.strip():
                def _skip(sid=sid, cva=speaker["cv_created_at"]):
                    with get_conn() as conn:
                        save_annotation(conn, sid, GROUP, None, SKIPPED,
                                        config.MODEL_ANNOTATE_D,
                                        config.PROMPT_VERSION_ANNOTATE_D,
                                        cv_created_at=cva, error="empty CV text")
                        set_speaker_status(conn, sid, "annotate_d", SKIPPED)
                _db_write_with_retry(_skip)
                outcomes[SKIPPED] += 1
                continue

            try:
                result = annotate(cv_text, name)
                def _save(sid=sid, result=result, cva=speaker["cv_created_at"]):
                    with get_conn() as conn:
                        save_annotation(conn, sid, GROUP, result, SUCCESS,
                                        config.MODEL_ANNOTATE_D,
                                        config.PROMPT_VERSION_ANNOTATE_D,
                                        cv_created_at=cva)
                        set_speaker_status(conn, sid, "annotate_d", SUCCESS,
                                           annotate_d_model=config.MODEL_ANNOTATE_D,
                                           annotate_d_prompt_v=config.PROMPT_VERSION_ANNOTATE_D)
                _db_write_with_retry(_save)
                outcomes[SUCCESS] += 1
            except Exception as e:
                err = str(e)[:500]
                def _fail(sid=sid, err=err, cva=speaker["cv_created_at"]):
                    with get_conn() as conn:
                        save_annotation(conn, sid, GROUP, None, FAILED,
                                        config.MODEL_ANNOTATE_D,
                                        config.PROMPT_VERSION_ANNOTATE_D,
                                        cv_created_at=cva, error=err)
                        set_speaker_status(conn, sid, "annotate_d", FAILED, error=err)
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
                    conn, "annotate_d", failed_ids,
                    name=f"annotate_d_failures_{run_id[:8]}",
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
