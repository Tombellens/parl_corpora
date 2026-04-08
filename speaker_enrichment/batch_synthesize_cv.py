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
)


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


def build_merge_prompt(name: str, snippets: list[dict]) -> str:
    parts = [f"Person: {name}\n\nBiographical snippets from {len(snippets)} web sources:\n"]
    for i, s in enumerate(snippets, start=1):
        url  = s["url"]
        lang = s["query_language"] or "?"
        text = (s["synthesis_text"] or "").strip()
        parts.append(f"--- Source {i} (lang={lang}, rank={s['search_rank']}) ---")
        parts.append(f"URL: {url}")
        parts.append(text)
        parts.append("")
    parts.append("Write the merged biographical CV:")
    return "\n".join(parts)


def process_speaker(conn, speaker: dict) -> bool:
    sid  = speaker["speaker_id"]
    name = speaker["name_cleaned"] or "Unknown"

    snippets = get_synthesised_snippets(conn, sid)

    if not snippets:
        set_speaker_status(conn, sid, "cv_synth", SKIPPED,
                           error="no synthesised URL snippets available")
        return True

    # Build prompt
    merge_prompt = build_merge_prompt(name, snippets)
    source_url_ids = [s["id"] for s in snippets]

    cv_text = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": merge_prompt},
        ],
        model=config.MODEL_SYNTHESIZE_CV,
        max_tokens=2048,
    )

    if not cv_text.strip():
        set_speaker_status(conn, sid, "cv_synth", FAILED, error="LLM returned empty CV")
        return False

    # Save CV to file (traceability for the paper)
    cv_dir  = Path(config.CV_DIR)
    cv_dir.mkdir(parents=True, exist_ok=True)
    cv_path = cv_dir / f"{sid}.txt"
    cv_path.write_text(cv_text, encoding="utf-8")

    save_cv(conn, sid, cv_text, str(cv_path), source_url_ids,
            model=config.MODEL_SYNTHESIZE_CV,
            prompt_version=config.PROMPT_VERSION_SYNTHESIZE_CV)

    set_speaker_status(conn, sid, "cv_synth", SUCCESS,
                       cv_synth_model=config.MODEL_SYNTHESIZE_CV,
                       cv_synth_prompt_v=config.PROMPT_VERSION_SYNTHESIZE_CV)
    return True


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

    try:
        acquire_llm_lock("synthesize_cv", config.MODEL_SYNTHESIZE_CV)
        print(f"Loading model {config.MODEL_SYNTHESIZE_CV}...")
        load_model(config.MODEL_SYNTHESIZE_CV)
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

        n_success = n_failed = 0
        failed_ids: list[str] = []

        for speaker in tqdm(speakers, desc="Merging CVs"):
            sid = speaker["speaker_id"]
            with get_conn() as conn:
                set_speaker_status(conn, sid, "cv_synth", "running")
            try:
                with get_conn() as conn:
                    process_speaker(conn, speaker)
                n_success += 1
            except Exception as e:
                with get_conn() as conn:
                    set_speaker_status(conn, sid, "cv_synth", FAILED, error=str(e)[:500])
                failed_ids.append(sid)
                n_failed += 1
                tqdm.write(f"  FAILED {speaker['name_cleaned']}: {e}")

        if failed_ids:
            with get_conn() as conn:
                fb_id = create_failure_batch(
                    conn, "cv_synth", failed_ids,
                    name=f"cv_synth_failures_{run_id[:8]}",
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
