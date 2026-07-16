"""
detect_targets.py
=================
Phase 1 target detection. Classify the TARGET of each detected accusation
(broad type + raw mention) with the local LLM — ONE accusation per call, full
reasoning.

Throughput comes from CONCURRENCY: the model is loaded with N parallel slots and
a thread pool fires N single-accusation requests at once. Crash/server recovery
in llm_client is concurrency-safe (coalesced behind a lock), so a model crash
during the run self-heals without a stampede of reloads.

Reuses the hardened LM Studio client from ../speaker_enrichment.

Usage:
    python3 detect_targets.py [--limit N] [--workers K]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import config
from db import get_conn, init_db, now_iso, SUCCESS, FAILED

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speaker_enrichment"))
from llm_client import (  # noqa: E402
    acquire_llm_lock, chat, extract_json,
    is_llm_locked, load_model, release_llm_lock, unload_model,
)


SYSTEM_PROMPT = """You identify the TARGET of an accusation of lying or untruth
made in parliamentary debate.

You are given a short excerpt of consecutive sentences from a debate. Lines are
grouped under the speaker who said them, shown in brackets like "[Maria Fekter]".
ONE sentence — the accusation — is marked with ">>>"; the speaker whose block it
sits in is the ACCUSER. The preceding lines (which may be an earlier speaker's
turn) are context. Decide to WHOM or WHAT the accusation (the ">>>" sentence) is
directed. The person speaking in the turn just before the accuser is often the
target (e.g. the accusation answers what they just said).

Return a single JSON object:
  {"target_type": "<type>", "target_text": "<mention>"}

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
- The target is WHO OR WHAT IS ACCUSED of lying / untruth / being wrong — not the
  audience being addressed or informed (e.g. "the House", "the public"), and not
  a person merely cited as evidence. If X "told the public the untruth", the
  target is X, not the public.
- Judge only the accusation sentence (marked ">>>"); use the other lines only to
  understand who/what it refers to.
- A name (or party) followed by a colon at the start of a line — e.g.
  "Dr. Fekter:" or "Wabl:" — marks the SPEAKER of that interjection, NOT the
  target. That person is the one making the accusation. When the accusation is
  such an interjection, the target is whoever/whatever it responds to, usually
  named in the preceding context — not the speaker before the colon.
- "target_text" = the target as written (e.g. "the government", "Russia",
  "the Social Democrats"); use "" for unclear_or_none.
- Base the answer only on the text. Do not guess a target that is not supported.
Respond ONLY with the JSON object, nothing else."""


def classify_one(row: dict) -> tuple[str, str]:
    """Return (target_type, target_text) for one accusation."""
    user_msg = (
        "Excerpt (the accusation is the line marked \">>>\"):\n\n"
        f"{row['context'] or row['sentence']}\n\n"
        "Classify the target of the accusation:"
    )
    response = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        model=config.MODEL,
        max_tokens=config.TARGET_MAX_TOKENS,
    )
    obj = extract_json(response)
    if not isinstance(obj, dict):
        raise ValueError("LLM did not return a JSON object")
    ttype = str(obj.get("target_type", "")).strip().lower()
    if ttype not in config.TARGET_TYPE_SET:
        ttype = "unclear_or_none"
    ttext = "" if ttype == "unclear_or_none" else str(obj.get("target_text") or "")[:300]
    return ttype, ttext


def _db_write_with_retry(fn, max_attempts: int = 12, base_delay: float = 0.5):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise


def _process(row: dict) -> bool:
    """Classify one accusation and persist the result. Returns True on success."""
    rid = row["id"]
    try:
        ttype, ttext = classify_one(row)
        def _save():
            with get_conn() as conn:
                conn.execute(
                    """UPDATE accusations
                       SET target_status=?, target_type=?, target_text=?,
                           target_at=?, target_model=?, target_prompt_v=?
                       WHERE id=?""",
                    (SUCCESS, ttype, ttext, now_iso(), config.MODEL,
                     config.PROMPT_VERSION, rid),
                )
        _db_write_with_retry(_save)
        return True
    except Exception as e:
        err = str(e)[:400]
        def _fail():
            with get_conn() as conn:
                conn.execute(
                    "UPDATE accusations SET target_status=?, target_error=?, target_at=? WHERE id=?",
                    (FAILED, err, now_iso(), rid),
                )
        try:
            _db_write_with_retry(_fail)
        except Exception:
            pass
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=config.N_WORKERS)
    args = ap.parse_args()

    if is_llm_locked():
        print("LLM is currently in use by another process. Exiting.")
        return

    init_db()
    with get_conn() as conn:
        conn.execute("UPDATE accusations SET target_status='pending' WHERE target_status='running'")
        total_pending = conn.execute(
            "SELECT COUNT(*) FROM accusations WHERE target_status='pending'"
        ).fetchone()[0]
    remaining = total_pending if args.limit is None else min(args.limit, total_pending)
    print(f"{total_pending:,} pending  |  processing {remaining:,} this run  |  workers={args.workers}")

    _loaded = None
    n_ok = n_fail = done = 0
    t0 = time.time()
    try:
        acquire_llm_lock("detect_targets", config.MODEL)
        print(f"Loading model {config.MODEL} ...")
        _loaded = load_model(config.MODEL, context_length=config.LLM_CONTEXT_LENGTH,
                             num_parallel=config.LLM_NUM_PARALLEL).get("instance_id")

        pbar = tqdm(total=remaining, desc="Targets")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while done < remaining:
                take = min(config.FETCH_CHUNK, remaining - done)
                with get_conn() as conn:
                    rows = [dict(r) for r in conn.execute(
                        "SELECT * FROM accusations WHERE target_status='pending' LIMIT ?",
                        (take,),
                    ).fetchall()]
                if not rows:
                    break
                # Mark this chunk running so a crash/restart doesn't re-pull them.
                ids = [r["id"] for r in rows]
                with get_conn() as conn:
                    conn.execute(
                        f"UPDATE accusations SET target_status='running' "
                        f"WHERE id IN ({','.join('?'*len(ids))})", ids)

                futures = [pool.submit(_process, r) for r in rows]
                for fut in as_completed(futures):
                    ok = fut.result()
                    n_ok += ok
                    n_fail += (not ok)
                    done += 1
                    pbar.update(1)
                    rate = done / max(1e-9, time.time() - t0)
                    pbar.set_postfix(ok=n_ok, fail=n_fail, rps=f"{rate:.1f}")
        pbar.close()
    finally:
        if _loaded:
            try:
                unload_model(_loaded)
            except Exception:
                pass
        release_llm_lock()

    dt = time.time() - t0
    print(f"\nDone.  success={n_ok:,}  failed={n_fail:,}  "
          f"in {dt/60:.1f} min  ({done/max(1e-9,dt):.1f} accusations/s)")


if __name__ == "__main__":
    main()
