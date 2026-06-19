"""
batch_annotate_b.py
===================
Submodule 2 — Group B annotation: political party affiliations, anchored to
PartyFacts IDs.

Approach: instead of extracting free-text party names and fuzzy-linking to
PartyFacts afterwards, we give the model a CLOSED per-country list of valid
parties (PartyFacts id | abbrev | name | years), built by prepare_partyfacts.py
and stored in party_prompts.json. The model returns a chronological party
HISTORY using those ids, scoped to affiliations relevant after 1990. Parties
not in the list are recorded with partyfacts_id=null (escape hatch) so nothing
is force-fit. Every returned id is validated against the country's allowed set;
any id not in the list is nulled (and recorded) so stored ids are always real
PartyFacts ids.

Schema of annotation_json:
{
  "parties": [
    {
      "partyfacts_id":  36 | null,        // valid id from the country list, or null
      "partyfacts_name": "New Flemish Alliance (...)" | null,  // canonical, filled when id valid
      "party_name_raw": "N-VA",           // party as named/identifiable in the CV
      "start_year":     2003 | null,
      "end_year":       null              // null = current / unknown
    },
    ...
  ],
  "n_parties":   <int>,
  "confidence":  "high" | "medium" | "low",
  "invalid_ids_nulled": [ ... ]           // present only if the model returned bad ids
}

Usage:
    python3 batch_annotate_b.py [--limit N] [--failure-batch-id ID]
"""

import argparse
import json
import time
import uuid
from pathlib import Path

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


GROUP = "B"

SYSTEM_PROMPT = """You are a data extraction assistant for a scientific study on politicians.

You will be given:
1. The politician's country.
2. A list of valid political parties for that country, one per line, formatted:
   partyfacts_id | abbreviation | name (native name) | years_active
3. A biographical CV.

GOAL: determine which party the person belonged to at any given time during
their political career, so that a later step can map any date to the party they
belonged to at that moment. Return a single JSON object.

What counts as a party affiliation:
- Code an affiliation whenever the CV clearly associates the person with a
  party — through membership, holding party office, working for the party, or
  serving as that party's minister, member of parliament, or candidate.
- Use the provided list as the scope of parties to consider (it already covers
  the parties relevant to this period). Match by abbreviation, name, or native
  name.

History and dates:
- If the person belonged to ONE party throughout, return a single entry;
  start_year and end_year may be null when the CV does not clearly state them.
- If the person SWITCHED parties, return one entry per party in chronological
  order, and give the switch boundary years as well as the CV supports them
  (the old party's end_year and/or the new party's start_year). Dates matter
  mainly to mark these switch points.
- Do NOT invent precise years. Use the years stated in the CV; use null when a
  year is unclear. Approximate years that the CV states are fine.

Output JSON:
{
  "parties": [
    {
      "partyfacts_id": <integer id taken from the list, or null if the party is not in the list>,
      "party_name_raw": "<party as named or identifiable in the CV>",
      "start_year": <integer, or null>,
      "end_year": <integer, or null = current / unknown>
    }
  ],
  "n_parties": <integer>,
  "confidence": "high" | "medium" | "low"
}

Rules:
- Use ONLY partyfacts_id values that appear in the provided list. If a clearly
  associated party is NOT in the list, set partyfacts_id to null but still fill
  party_name_raw.
- Do NOT fabricate an affiliation that has no basis in the CV.
- If the CV gives no basis for any party affiliation, return
  {"parties": [], "n_parties": 0, "confidence": "high"}.
Respond ONLY with the JSON object."""


def load_party_data() -> dict:
    """Load party_prompts.json -> {iso2: {"parties":[...], "prompt_block": "..."}}."""
    path = Path(config.PARTY_PROMPTS_JSON)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_partyfacts.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def annotate(cv_text: str, name: str, country: str, country_data: dict) -> dict:
    """
    Extract the post-1990 party history for one speaker, anchored to the
    country's PartyFacts list. Validates returned ids against the allowed set.
    """
    prompt_block = country_data["prompt_block"]
    valid_ids = {p["partyfacts_id"] for p in country_data["parties"]}
    id_to_name = {p["partyfacts_id"]: p["name"] for p in country_data["parties"]}

    user_msg = (
        f"Country: {country}\n\n"
        f"Valid parties for {country} (partyfacts_id | abbrev | name | years):\n"
        f"{prompt_block}\n\n"
        f"Person: {name}\n\n"
        f"CV:\n{cv_text}\n\n"
        "Extract the post-1990 party history using the PartyFacts IDs above:"
    )

    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL_ANNOTATE_B,
        max_tokens=1024,
    )
    result = extract_json(response)

    parties = result.get("parties") or []
    nulled = []
    for p in parties:
        pid = p.get("partyfacts_id")
        pid_int = None
        if pid is not None:
            try:
                pid_int = int(pid)
            except (ValueError, TypeError):
                pid_int = None
        if pid_int is not None and pid_int in valid_ids:
            p["partyfacts_id"]  = pid_int
            p["partyfacts_name"] = id_to_name.get(pid_int)
        else:
            if pid is not None:
                nulled.append(pid)        # model returned an id not in the list
            p["partyfacts_id"]   = None
            p["partyfacts_name"] = None
        p.setdefault("party_name_raw", None)
        p.setdefault("start_year", None)
        p.setdefault("end_year", None)

    result["parties"] = parties
    result["n_parties"] = len(parties)
    result.setdefault("confidence", None)
    if nulled:
        result["invalid_ids_nulled"] = nulled
    return result


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

    party_data = load_party_data()
    print(f"Loaded PartyFacts lists for {len(party_data)} countries")

    init_db()
    run_id = str(uuid.uuid4())

    _loaded_instance = None
    try:
        acquire_llm_lock(f"annotate_{GROUP}", config.MODEL_ANNOTATE_B)
        print(f"Loading model {config.MODEL_ANNOTATE_B}...")
        _loaded_instance = load_model(
            config.MODEL_ANNOTATE_B, context_length=config.LLM_CONTEXT_LENGTH
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
                         AND s.annotate_b_status = 'pending'
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
            country = speaker["country"]

            with get_conn() as conn:
                set_speaker_status(conn, sid, "annotate_b", "running")

            # Guard: no CV, or no PartyFacts list for this country -> skip cleanly
            skip_reason = None
            if not cv_text.strip():
                skip_reason = "empty CV text"
            elif country not in party_data:
                skip_reason = f"no PartyFacts list for country {country!r}"

            if skip_reason:
                def _skip(sid=sid, reason=skip_reason, cva=speaker["cv_created_at"]):
                    with get_conn() as conn:
                        save_annotation(conn, sid, GROUP, None, SKIPPED,
                                        config.MODEL_ANNOTATE_B,
                                        config.PROMPT_VERSION_ANNOTATE_B,
                                        cv_created_at=cva, error=reason)
                        set_speaker_status(conn, sid, "annotate_b", SKIPPED, error=reason)
                _db_write_with_retry(_skip)
                outcomes[SKIPPED] += 1
                continue

            try:
                result = annotate(cv_text, name, country, party_data[country])
                def _save(sid=sid, result=result, cva=speaker["cv_created_at"]):
                    with get_conn() as conn:
                        save_annotation(conn, sid, GROUP, result, SUCCESS,
                                        config.MODEL_ANNOTATE_B,
                                        config.PROMPT_VERSION_ANNOTATE_B,
                                        cv_created_at=cva)
                        set_speaker_status(conn, sid, "annotate_b", SUCCESS,
                                           annotate_b_model=config.MODEL_ANNOTATE_B,
                                           annotate_b_prompt_v=config.PROMPT_VERSION_ANNOTATE_B)
                _db_write_with_retry(_save)
                outcomes[SUCCESS] += 1
            except Exception as e:
                err = str(e)[:500]
                def _fail(sid=sid, err=err, cva=speaker["cv_created_at"]):
                    with get_conn() as conn:
                        save_annotation(conn, sid, GROUP, None, FAILED,
                                        config.MODEL_ANNOTATE_B,
                                        config.PROMPT_VERSION_ANNOTATE_B,
                                        cv_created_at=cva, error=err)
                        set_speaker_status(conn, sid, "annotate_b", FAILED, error=err)
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
                    conn, "annotate_b", failed_ids,
                    name=f"annotate_b_failures_{run_id[:8]}",
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
