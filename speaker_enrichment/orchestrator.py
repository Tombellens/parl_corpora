"""
orchestrator.py
===============
Checks pipeline state and decides what to run next, enforcing the LLM mutex.

Logic:
  1. If the LLM lock is held by a live process → nothing to schedule, exit.
  2. Check in order: url_synth → cv_synth → annotate_a/b/c/d
     Run whichever LLM stage has the most pending speakers ready.
  3. Non-LLM stages (query, fetch) are always safe to run; the orchestrator
     can trigger them independently as background processes.

Can be called from cron or the dashboard.

Usage:
    python3 orchestrator.py [--dry-run] [--force-stage STAGE]
    # Typical cron (every 2 hours):
    # 0 */2 * * * cd /home/tom/parl_corpora/notebooks/speaker_enrichment && \
    #   python3 orchestrator.py >> /home/tom/data/speaker_enrichment/logs/orchestrator.log 2>&1
"""

import argparse
import subprocess
import sys
from pathlib import Path

from db import get_conn, init_db, STAGES
from llm_client import is_llm_locked, read_llm_lock

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Stage readiness queries
# ---------------------------------------------------------------------------

STAGE_READY_SQL = {
    "query": "SELECT COUNT(*) FROM speakers WHERE query_status = 'pending'",
    "fetch": (
        "SELECT COUNT(*) FROM speakers "
        "WHERE query_status = 'success' AND fetch_status = 'pending'"
    ),
    "url_synth": (
        "SELECT COUNT(*) FROM speakers "
        "WHERE fetch_status = 'success' AND url_synth_status = 'pending'"
    ),
    "cv_synth": (
        "SELECT COUNT(*) FROM speakers "
        "WHERE url_synth_status IN ('success','skipped') AND cv_synth_status = 'pending'"
    ),
    "annotate_a": (
        "SELECT COUNT(*) FROM speakers "
        "WHERE cv_synth_status = 'success' AND annotate_a_status = 'pending'"
    ),
    "annotate_b": (
        "SELECT COUNT(*) FROM speakers "
        "WHERE cv_synth_status = 'success' AND annotate_b_status = 'pending'"
    ),
    "annotate_c": (
        "SELECT COUNT(*) FROM speakers "
        "WHERE cv_synth_status = 'success' AND annotate_c_status = 'pending'"
    ),
    "annotate_d": (
        "SELECT COUNT(*) FROM speakers "
        "WHERE cv_synth_status = 'success' AND annotate_d_status = 'pending'"
    ),
}

STAGE_SCRIPT = {
    "query":      "batch_query.py",
    "fetch":      "batch_fetch.py",
    "url_synth":  "batch_synthesize_url.py",
    "cv_synth":   "batch_synthesize_cv.py",
    "annotate_a": "batch_annotate_a.py",
    "annotate_b": "batch_annotate_b.py",
    "annotate_c": "batch_annotate_c.py",
    "annotate_d": "batch_annotate_d.py",
}

# Stages that require the LLM (and thus the lock)
LLM_STAGES = {"url_synth", "cv_synth", "annotate_a", "annotate_b", "annotate_c", "annotate_d"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_ready_counts() -> dict[str, int]:
    with get_conn() as conn:
        return {
            stage: conn.execute(sql).fetchone()[0]
            for stage, sql in STAGE_READY_SQL.items()
        }


def launch(stage: str, dry_run: bool = False) -> None:
    script = HERE / STAGE_SCRIPT[stage]
    cmd    = [sys.executable, str(script)]
    print(f"  → Launching: {' '.join(cmd)}")
    if not dry_run:
        subprocess.Popen(cmd, cwd=str(HERE))


# ---------------------------------------------------------------------------
# Main decision logic
# ---------------------------------------------------------------------------

def decide(dry_run: bool = False, force_stage: str | None = None) -> None:
    init_db()
    ready = get_ready_counts()

    print("Pipeline readiness:")
    for stage, count in ready.items():
        llm = " [LLM]" if stage in LLM_STAGES else ""
        print(f"  {stage:<14} {count:>6} pending{llm}")

    if force_stage:
        print(f"\nForcing stage: {force_stage}")
        if force_stage in LLM_STAGES and is_llm_locked():
            lock = read_llm_lock()
            print(f"  Cannot force — LLM locked by PID {lock['pid']} ({lock['task']})")
            return
        launch(force_stage, dry_run)
        return

    # Always launch non-LLM stages if they have work
    for stage in ("query", "fetch"):
        if ready[stage] > 0:
            print(f"\nNon-LLM stage '{stage}' has {ready[stage]} pending → launching")
            launch(stage, dry_run)

    # For LLM stages: check lock first
    if is_llm_locked():
        lock = read_llm_lock()
        print(f"\nLLM locked by PID {lock['pid']} task={lock['task']} — skipping LLM stages")
        return

    # Pick the LLM stage with the most work ready (greedy)
    llm_ready = {s: ready[s] for s in LLM_STAGES}
    best_stage = max(llm_ready, key=llm_ready.get)

    if llm_ready[best_stage] == 0:
        print("\nNo LLM stages have pending work.")
        return

    print(f"\nLLM stage '{best_stage}' has {llm_ready[best_stage]} pending → launching")
    launch(best_stage, dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be launched, don't actually launch")
    parser.add_argument("--force-stage", choices=list(STAGE_SCRIPT.keys()),
                        help="Force-launch a specific stage regardless of readiness order")
    args = parser.parse_args()

    decide(dry_run=args.dry_run, force_stage=args.force_stage)


if __name__ == "__main__":
    main()
