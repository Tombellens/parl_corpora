"""
batch_annotate_d.py
===================
Submodule 2 — Group D annotation: sectors of professional experience.

Simplified design: instead of a full career timeline (titles, organisations,
dates), we record only the SET of high-level sectors in which the person had
professional / occupational experience. Sectors are the first-digit groups
(1-8) of the political-elite career codebook. No durations, no titles — just a
deduplicated set of sector code numbers.

Schema of annotation_json:
{
  "sectors": [1, 4, 6],                             // deduplicated first-digit codes
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

# High-level (first-digit) sector groups from the political-elite career
# codebook. We record only the top-level group a position belongs to.
SECTOR_CODES = {1, 2, 3, 4, 5, 6, 7, 8}

SYSTEM_PROMPT = """You are a data extraction assistant for a scientific study on politicians.

From a biographical CV, identify the SECTORS in which the person had professional
or occupational experience at any point in their career. Return a single JSON
object with a deduplicated set of high-level sector codes. Do NOT report job
titles, organisations, durations, or dates — only the set of sector code numbers.

Assign each position to one of these high-level sectors and return the set of
sector numbers that apply:

1 = Central government executive ("executive triangle"): head of state or
    government and ministers (president, prime minister, deputy PM, minister with
    or without portfolio, European Commissioner), and senior political, advisory
    or top-bureaucratic roles inside ministries, the centre of government, or the
    presidential office (e.g. head of ministerial cabinet, secretary/director
    general).
2 = Public administration (outside the executive triangle): civil servants and
    bureaucrats at national, sub-national and local level; heads and staff of
    non-ministerial public bodies; prefects/governors; non-political
    administration of parliament.
    NOTE: this is administrative / civil-service work only. Public-sector
    TEACHERS and academics, DOCTORS and health workers, and public
    BROADCASTERS belong to 3, NOT 2.
3 = Public sector organisations and industries: state-owned or public bodies and
    enterprises — finance/central bank, transport, utilities, telecom/postal,
    public broadcasting, public healthcare, public education and research (school,
    university, other), police, military and defence, diplomatic service.
4 = Politics / political office and employment (outside the executive triangle):
    executive office such as mayor or regional minister; legislative office such
    as MEP, senator, regional or local councillor; party office/functions; paid
    political employment (for a party, MP or MEP); work for party foundations or
    politically affiliated organisations.
    IMPORTANT: do NOT count the person's own seat in the NATIONAL parliament
    (being an MP / member of the national legislature) — that is true of everyone
    in this study and must be excluded. Still assign 4 for OTHER political roles
    (mayor, regional/local councillor, MEP, party office, political employment).
5 = Judiciary and oversight: prosecutors, judges, public-sector lawyers/notaries,
    court and judiciary administration, constitutional court, accountability
    institutions (ombudsperson, audit, anti-corruption, etc.), regulatory agencies.
6 = Private and third sector: private-sector managers, employees and self-employed
    (at home or abroad); NGOs, trade unions, associations, foundations and
    charities; non-public media (journalist); religious bodies.
7 = International organisations (non-political): administrative roles in the EU
    institutions, UN, NATO, World Bank and other international organisations.
8 = Other: unemployed, retired, prison, no prior job, parental leave, or a period
    of further education / career break.

Rules:
- Include a sector number if the CV shows the person worked in it at any point.
- Assign a sector only for a SUBSTANTIVE job or occupation the person actually
  held. Do NOT assign a sector merely for an honorary position, a board or
  committee membership, a patronage, a decoration, or a ceremonial / one-off
  role.
- Do NOT code the person's own national parliamentary mandate (being an MP /
  member of the national parliament). It is constant across everyone and must be
  excluded. Other, genuine political office (mayor, councillor, MEP, party
  functionary, political employment) still counts under 4.
- Base sectors only on what the CV states. Do NOT infer experience with no basis.
- Return each number at most once.
- If the CV shows no codeable professional experience outside the national
  parliamentary mandate, return {"sectors": [], "confidence": "high"}.

Output JSON:
{
  "sectors": [1, 4, 6],
  "confidence": "high" | "medium" | "low"
}
Respond ONLY with the JSON object."""


def annotate(cv_text: str, name: str) -> dict:
    user_msg = f"Person: {name}\n\nCV:\n{cv_text}\n\nList Group D (sector code numbers):"
    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_ANNOTATE_D,
        max_tokens=1024,   # headroom for reasoning-model output before the JSON
    )
    result = extract_json(response)

    # Validate against the codebook {1..8}, deduplicate, sort.
    found = set()
    for s in (result.get("sectors") or []):
        try:
            code = int(s)
        except (ValueError, TypeError):
            continue
        if code in SECTOR_CODES:
            found.add(code)
    sectors = sorted(found)

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
