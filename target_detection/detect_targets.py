"""
detect_targets.py
=================
Phase 1 target detection. For each detected accusation, classify the TARGET of
the accusation (broad type + raw mention) using the local LLM.

Throughput comes from batching many accusations per LLM call (their contexts are
tiny), run sequentially so the hardened crash-recovery in llm_client stays valid.

Reuses the hardened LM Studio client from ../speaker_enrichment (lms-CLI load,
crash/server auto-recovery, robust JSON extraction).

Usage:
    python3 detect_targets.py [--limit N] [--batch K]
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

from tqdm import tqdm

import config
from db import get_conn, init_db, now_iso, SUCCESS, FAILED

# Reuse the hardened LLM client from the speaker_enrichment module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speaker_enrichment"))
from llm_client import (  # noqa: E402
    acquire_llm_lock, chat, extract_json,
    is_llm_locked, load_model, release_llm_lock, unload_model,
)


SYSTEM_PROMPT = """You identify the TARGET of accusations of lying or untruth made
in parliamentary debate.

You are given a numbered list of items. Each item is a short excerpt from a
speech in which ONE sentence — the accusation — is marked with ">>>". The
surrounding lines are context only. For each item, decide to WHOM or WHAT the
accusation (the ">>>" sentence) is directed.

Return a JSON array with one object per item, in the same order, each:
  {"n": <item number>, "target_type": "<type>", "target_text": "<mention>"}

"target_type" must be EXACTLY one of:
  "person"                  : a specific named individual
  "government"              : a government function holder (minister, PM, ...) or the government as a whole
  "administration"          : civil service / bureaucracy / a ministry as an administrative apparatus
  "political_party"         : a party or parliamentary group
  "public_institution"      : courts, central bank, agencies, army/police, other state bodies
  "foreign_country_or_govt" : another country or its government
  "international_org"        : EU, UN, NATO, and other international organisations
  "media"                   : press, broadcasters, journalists
  "social_group"            : a demographic / social / ethnic / religious group
  "other"                   : a real target that fits none of the above
  "unclear_or_none"         : no identifiable target, or purely rhetorical

Rules:
- Judge only the accusation sentence (marked ">>>"); use the other lines only to
  understand who/what it refers to.
- "target_text" = the target as written (e.g. "the government", "Russia",
  "Mr Sch<C3><BC>ssel", "the Social Democrats"); use "" for unclear_or_none.
- Base the answer only on the text. Do not guess a target that is not supported.
Respond ONLY with the JSON array, nothing else."""


def _build_user_msg(items: list[dict]) -> str:
    lines = []
    for n, row in enumerate(items, start=1):
        lines.append(f"[{n}]")
        lines.append(row["context"] or row["sentence"] or "")
        lines.append("")
    return "Classify the target of each accusation:\n\n" + "\n".join(lines)


def _classify_batch(items: list[dict]) -> dict[int, tuple[str, str]]:
    """Return {item_number(1-based): (target_type, target_text)} from the LLM."""
    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_msg(items)},
        ],
        model=config.MODEL,
        max_tokens=config.TARGET_MAX_TOKENS,
    )
    results = extract_json(response)
    if not isinstance(results, list):
        raise ValueError("LLM did not return a JSON array")

    out: dict[int, tuple[str, str]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            n = int(r.get("n"))
        except (TypeError, ValueError):
            continue
        ttype = str(r.get("target_type", "")).strip().lower()
        if ttype not in config.TARGET_TYPE_SET:
            ttype = "unclear_or_none"          # coerce unrecognised types
        ttext = r.get("target_text") or ""
        if ttype == "unclear_or_none":
            ttext = ""
        out[n] = (ttype, str(ttext)[:300])
    return out


def _db_write_with_retry(fn, max_attempts: int = 10, base_delay: float = 1.0):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Max accusations to process this run (default: all pending)")
    ap.add_argument("--batch", type=int, default=config.TARGET_BATCH_ACCUSATIONS)
    args = ap.parse_args()

    if is_llm_locked():
        print("LLM is currently in use by another process. Exiting.")
        return

    init_db()
    # Recover any rows stuck 'running' from a previous interrupted run.
    with get_conn() as conn:
        conn.execute("UPDATE accusations SET target_status='pending' WHERE target_status='running'")
        total_pending = conn.execute(
            "SELECT COUNT(*) FROM accusations WHERE target_status='pending'"
        ).fetchone()[0]
    remaining = total_pending if args.limit is None else min(args.limit, total_pending)
    print(f"{total_pending:,} pending  |  processing {remaining:,} this run  |  batch={args.batch}")

    _loaded = None
    n_success = n_failed = 0
    t0 = time.time()
    try:
        acquire_llm_lock("detect_targets", config.MODEL)
        print(f"Loading model {config.MODEL} ...")
        _loaded = load_model(config.MODEL, context_length=config.LLM_CONTEXT_LENGTH,
                             num_parallel=config.LLM_NUM_PARALLEL).get("instance_id")

        pbar = tqdm(total=remaining, desc="Targets")
        done = 0
        while done < remaining:
            take = min(args.batch, remaining - done)
            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM accusations WHERE target_status='pending' LIMIT ?",
                    (take,),
                ).fetchall()
            if not rows:
                break
            items = [dict(r) for r in rows]

            try:
                mapping = _classify_batch(items)
            except Exception as e:
                mapping = {}
                tqdm.write(f"  batch failed: {str(e)[:160]}")

            ts = now_iso()
            def _save(items=items, mapping=mapping, ts=ts):
                with get_conn() as conn:
                    for n, row in enumerate(items, start=1):
                        if n in mapping:
                            ttype, ttext = mapping[n]
                            conn.execute(
                                """UPDATE accusations
                                   SET target_status=?, target_type=?, target_text=?,
                                       target_at=?, target_model=?, target_prompt_v=?
                                   WHERE id=?""",
                                (SUCCESS, ttype, ttext, ts, config.MODEL,
                                 config.PROMPT_VERSION, row["id"]),
                            )
                        else:
                            conn.execute(
                                """UPDATE accusations
                                   SET target_status=?, target_error=?, target_at=?
                                   WHERE id=?""",
                                (FAILED, "no result in batch response", ts, row["id"]),
                            )
            _db_write_with_retry(_save)

            got = sum(1 for n in range(1, len(items) + 1) if n in mapping)
            n_success += got
            n_failed  += len(items) - got
            done      += len(items)
            pbar.update(len(items))
            rate = done / max(1e-9, time.time() - t0)
            pbar.set_postfix(ok=n_success, fail=n_failed, rps=f"{rate:.1f}")
        pbar.close()

    finally:
        if _loaded:
            try:
                unload_model(_loaded)
            except Exception:
                pass
        release_llm_lock()

    dt = time.time() - t0
    print(f"\nDone.  success={n_success:,}  failed={n_failed:,}  "
          f"in {dt/60:.1f} min  ({done/max(1e-9,dt):.1f} accusations/s)")


if __name__ == "__main__":
    main()
