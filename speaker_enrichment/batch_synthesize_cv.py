"""
batch_synthesize_cv.py
======================
Submodule 1, Part 4 — CV merger.

Takes all per-URL synthesis snippets for a speaker and asks the LLM to
produce one coherent, deduplicated biographical CV text.  This is the
end product of Submodule 1 and the input to all Submodule 2 annotators.

The CV is stored in:
  - speaker_cvs table (cv_text field)
  - CV_DIR/<speaker_id>.txt file (for direct traceability)
  - source_url_ids recorded (which URL snippets were merged)

Acquires LLM lock.

Usage:
    python3 batch_synthesize_cv.py [--limit N] [--failure-batch-id ID]
"""

import argparse
import time
import uuid
from pathlib import Path

from tqdm import tqdm

import config
from db import (
    FAILED, SUCCESS, SKIPPED,
    activate_failure_batch, create_failure_batch,
    fetch_pending_speakers, get_conn, init_db,
    now_iso, save_cv, set_speaker_status,
    get_synthesised_snippets,
)
from llm_client import (
    acquire_llm_lock, chat,
    is_llm_locked, load_model, release_llm_lock,
    unload_model,
)


# ---------------------------------------------------------------------------
# Context-length budgeting (same approach as batch_synthesize_url)
# ---------------------------------------------------------------------------
_OUTPUT_TOKENS_RESERVED = 2048
_PROMPT_OVERHEAD_TOKENS = 1200
_BYTES_PER_TOKEN        = 3.0
_MAX_INPUT_TOKENS       = 28000   # stay well below context for runtime stability


def _max_input_bytes() -> int:
    ctx_token_budget = (
        config.LLM_CONTEXT_LENGTH - _OUTPUT_TOKENS_RESERVED - _PROMPT_OVERHEAD_TOKENS
    )
    return max(0, int(min(_MAX_INPUT_TOKENS, ctx_token_budget) * _BYTES_PER_TOKEN))


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


SYSTEM_PROMPT = """You are a research assistant compiling a politician's biography
for a scientific study on parliamentary speech.

You will receive several biographical snippets about the same person, each
extracted from a different web source.  Your task is to merge them into one
clean, comprehensive biographical CV.

Rules:
- Include ALL factual information present across the snippets
- Remove exact duplicates (same fact mentioned multiple times → keep once)
- When sources contradict each other, note the discrepancy briefly
  (e.g. "born 1961 (one source says 1962)")
- Preserve approximate years wherever mentioned
- Write in plain prose, organised loosely as: personal info → education → career
- Do NOT add information not present in the snippets
- Do NOT use bullet points or headers — flowing prose only"""


def build_merge_prompt(name: str, snippets: list[dict]) -> tuple[str, list[int], int]:
    """
    Build the merge prompt, including snippets in rank order until the input
    byte budget is reached. Returns (prompt, used_url_ids, n_dropped).
    Snippets arrive ordered by search_rank (best first), so if we must drop
    any to fit the context, we drop the lowest-ranked.
    """
    budget = _max_input_bytes()
    header = f"Person: {name}\n\nBiographical snippets from web sources:\n"
    parts = [header]
    used_ids: list[int] = []
    running = len(header.encode("utf-8"))
    n_dropped = 0

    for i, s in enumerate(snippets, start=1):
        lang = s["query_language"] or "?"
        text = (s["synthesis_text"] or "").strip()
        block = (f"--- Source {i} (lang={lang}, rank={s['search_rank']}) ---\n"
                 f"URL: {s['url']}\n{text}\n")
        b = len(block.encode("utf-8"))
        if running + b > budget and used_ids:
            n_dropped = len(snippets) - len(used_ids)
            break
        parts.append(block)
        used_ids.append(s["id"])
        running += b

    parts.append("\nWrite the merged biographical CV:")
    return "\n".join(parts), used_ids, n_dropped


def process_speaker(speaker: dict) -> str:
    """
    Merge one speaker's URL snippets into a CV.

    The LLM call is made WITHOUT holding a DB connection (a multi-second call
    must not keep a write transaction open). DB reads/writes each use their own
    short transaction with lock retry. Returns the final status string.
    """
    sid  = speaker["speaker_id"]
    name = speaker["name_cleaned"] or "Unknown"

    with get_conn() as conn:
        snippets = get_synthesised_snippets(conn, sid)

    if not snippets:
        def _skip():
            with get_conn() as conn:
                set_speaker_status(conn, sid, "cv_synth", SKIPPED,
                                   error="no synthesised URL snippets available")
        _db_write_with_retry(_skip)
        return SKIPPED

    merge_prompt, source_url_ids, n_dropped = build_merge_prompt(name, snippets)
    note = f"dropped {n_dropped} low-rank snippets to fit context" if n_dropped else None

    # LLM call — no DB connection held here.
    cv_text = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": merge_prompt},
        ],
        model=config.MODEL_SYNTHESIZE_CV,
        max_tokens=_OUTPUT_TOKENS_RESERVED,
    )

    if not cv_text.strip():
        def _fail():
            with get_conn() as conn:
                set_speaker_status(conn, sid, "cv_synth", FAILED, error="LLM returned empty CV")
        _db_write_with_retry(_fail)
        return FAILED

    # Save CV to file (traceability for the paper)
    cv_dir = Path(config.CV_DIR)
    cv_dir.mkdir(parents=True, exist_ok=True)
    cv_path = cv_dir / f"{sid}.txt"
    cv_path.write_text(cv_text, encoding="utf-8")

    def _save():
        with get_conn() as conn:
            save_cv(conn, sid, cv_text, str(cv_path), source_url_ids,
                    model=config.MODEL_SYNTHESIZE_CV,
                    prompt_version=config.PROMPT_VERSION_SYNTHESIZE_CV)
            set_speaker_status(conn, sid, "cv_synth", SUCCESS,
                               error=note,
                               cv_synth_model=config.MODEL_SYNTHESIZE_CV,
                               cv_synth_prompt_v=config.PROMPT_VERSION_SYNTHESIZE_CV)
    _db_write_with_retry(_save)
    return SUCCESS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=config.BATCH_SIZE_SYNTHESIZE_CV)
    parser.add_argument("--failure-batch-id", type=int, default=None)
    args = parser.parse_args()

    if is_llm_locked():
        print("LLM is currently in use. Exiting.")
        return

    init_db()
    run_id = str(uuid.uuid4())

    _loaded_instance = None
    try:
        acquire_llm_lock("synthesize_cv", config.MODEL_SYNTHESIZE_CV)
        print(f"Loading model {config.MODEL_SYNTHESIZE_CV}...")
        _loaded_instance = load_model(config.MODEL_SYNTHESIZE_CV, context_length=config.LLM_CONTEXT_LENGTH).get("instance_id")
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO batch_runs (run_id, stage, batch_type, started_at) VALUES (?,?,?,?)",
                (run_id, "cv_synth",
                 "failure_retry" if args.failure_batch_id else "normal",
                 now_iso()),
            )

        with get_conn() as conn:
            if args.failure_batch_id:
                failure_ids = activate_failure_batch(conn, args.failure_batch_id)
                speakers    = fetch_pending_speakers(conn, "cv_synth", args.limit,
                                                     failure_ids=failure_ids)
            else:
                speakers = conn.execute(
                    """SELECT * FROM speakers
                       WHERE url_synth_status IN ('success','skipped')
                         AND cv_synth_status = 'pending'
                       LIMIT ?""",
                    (args.limit,),
                ).fetchall()

        print(f"Run {run_id[:8]}  |  {len(speakers)} speakers")

        outcomes = {SUCCESS: 0, FAILED: 0, SKIPPED: 0}
        n_errors = 0
        failed_ids: list[str] = []

        for speaker in tqdm(speakers, desc="Merging CVs"):
            sid = speaker["speaker_id"]
            with get_conn() as conn:
                set_speaker_status(conn, sid, "cv_synth", "running")
            try:
                status = process_speaker(speaker)
                outcomes[status] = outcomes.get(status, 0) + 1
                if status == FAILED:
                    failed_ids.append(sid)
            except Exception as e:
                # LLM/save error after retries — reset to pending so a rerun
                # retries this speaker rather than burying it.
                try:
                    with get_conn() as conn:
                        set_speaker_status(conn, sid, "cv_synth", "pending", error=str(e)[:500])
                except Exception:
                    pass
                failed_ids.append(sid)
                n_errors += 1
                tqdm.write(f"  ERROR {speaker['name_cleaned']}: {e}")

        if failed_ids:
            with get_conn() as conn:
                fb_id = create_failure_batch(
                    conn, "cv_synth", failed_ids,
                    name=f"cv_synth_failures_{run_id[:8]}",
                )
            print(f"  {len(failed_ids)} failures → failure_batch id={fb_id}")

        with get_conn() as conn:
            conn.execute(
                """UPDATE batch_runs
                   SET finished_at=?, n_attempted=?, n_success=?, n_failed=?
                   WHERE run_id=?""",
                (now_iso(), len(speakers), outcomes[SUCCESS],
                 outcomes[FAILED] + n_errors, run_id),
            )

        print(f"\nDone.  CV success={outcomes[SUCCESS]}  "
              f"skipped={outcomes[SKIPPED]}  failed={outcomes[FAILED]}  "
              f"finalise-errors={n_errors}")

    finally:
        if _loaded_instance:
            try:
                unload_model(_loaded_instance)
            except Exception:
                pass
        release_llm_lock()


if __name__ == "__main__":
    main()
