"""
resolve_targets.py
==================
Phase 2: resolve person-type accusation targets to actual speakers.

Tiered, reusing the interjection matcher (same country + active-period + surname):

  1. exact unique surname match  -> resolved deterministically (no LLM)
  2. exact but AMBIGUOUS (several same-surname MPs active then)
                                  -> LLM picks the right one from the context
  3. FUZZY (spelling/translation variant, e.g. Novotny -> Nowotny)
                                  -> LLM confirms it refers to that MP (or none)
  4. no exact/fuzzy candidate     -> unresolved
  5. no parseable surname (role/pronoun only, e.g. "the Minister")
                                  -> no_surname

The LLM only runs on the ambiguous + fuzzy tail. It sees the accusation context
(speaker-labelled) + the target as written + the candidate MPs (name, years) and
returns the candidate number, or 0 for none.

Usage:
    python3 resolve_targets.py [--limit N] [--workers K]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm

import config
from db import get_conn, init_db, now_iso

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speaker_enrichment"))
from llm_client import (  # noqa: E402
    acquire_llm_lock, chat, extract_json,
    is_llm_locked, load_model, release_llm_lock, unload_model,
)
import build_interjections as B  # reuse _surname, _load_speaker_index  # noqa: E402

FUZZY_THRESHOLD = 0.82
MAX_CANDIDATES  = 6

NEW_COLUMNS = {
    "resolve_method": "TEXT",   # exact | llm_ambiguous | llm_fuzzy
    "resolved_name":  "TEXT",   # name_cleaned of the resolved speaker
}


def _ensure_columns():
    with get_conn() as conn:
        have = {r[1] for r in conn.execute("PRAGMA table_info(accusations)").fetchall()}
        for col, decl in NEW_COLUMNS.items():
            if col not in have:
                conn.execute(f"ALTER TABLE accusations ADD COLUMN {col} {decl}")


def _candidates(surname, country, dataset, date, by_parl):
    """Return (kind, [(speaker_id, name), ...]).
    kind in {'exact','fuzzy','none'} — exact preferred; fuzzy = spelling variants."""
    cands = by_parl.get((country, dataset), [])
    d = (date or "")[:10]

    def in_period(lo, hi):
        return not (d and lo and hi) or (lo[:10] <= d <= hi[:10])

    exact = [(sid, name) for sid, name, sn, lo, hi in cands
             if sn == surname and in_period(lo, hi)]
    if exact:
        return "exact", exact

    fuzzy = []
    for sid, name, sn, lo, hi in cands:
        if not sn or not in_period(lo, hi):
            continue
        r = SequenceMatcher(None, surname, sn).ratio()
        if r >= FUZZY_THRESHOLD:
            fuzzy.append((r, sid, name))
    fuzzy.sort(key=lambda x: x[0], reverse=True)
    if fuzzy:
        return "fuzzy", [(sid, name) for _, sid, name in fuzzy[:MAX_CANDIDATES]]
    return "none", []


# ---------------------------------------------------------------------------
# LLM disambiguation / confirmation (ambiguous + fuzzy tail only)
# ---------------------------------------------------------------------------
LLM_SYSTEM = """You match the TARGET of an accusation of lying to the correct
member of parliament.

You are given: a debate excerpt (the accusation is the line marked ">>>"), the
TARGET as written in that accusation, and a numbered list of candidate MPs (name
and years active). Pick the ONE candidate that the target refers to. The written
name may be a spelling or translation variant of a candidate's name.

Answer with a single JSON object: {"choice": <candidate number>} — or
{"choice": 0} if none of the candidates is the target.
Respond ONLY with the JSON object."""


def _llm_pick(context, target_text, candidates) -> int:
    lines = [f"{i}. {name}" for i, (sid, name) in enumerate(candidates, 1)]
    user = (
        f"Excerpt (accusation marked \">>>\"):\n{context}\n\n"
        f"Target as written: {target_text!r}\n\n"
        f"Candidate MPs:\n" + "\n".join(lines) + "\n\n"
        "Which candidate is the target? Reply {\"choice\": N} or {\"choice\": 0}."
    )
    resp = chat(
        messages=[{"role": "system", "content": LLM_SYSTEM},
                  {"role": "user", "content": user}],
        model=config.MODEL, max_tokens=config.TARGET_MAX_TOKENS,
    )
    obj = extract_json(resp)
    if isinstance(obj, dict):
        try:
            return int(obj.get("choice"))
        except (TypeError, ValueError):
            return 0
    return 0


def _db_write_with_retry(fn, max_attempts=12, base_delay=0.5):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise


def _set(rid, status, speaker_id=None, name=None, method=None):
    def _w():
        with get_conn() as conn:
            conn.execute(
                "UPDATE accusations SET resolve_status=?, resolved_speaker_id=?, "
                "resolved_name=?, resolve_method=? WHERE id=?",
                (status, speaker_id, name, method, rid))
    _db_write_with_retry(_w)


def _resolve_one_llm(row, candidates, method):
    """LLM disambiguation/confirmation for one row."""
    try:
        choice = _llm_pick(row["context"] or row["sentence"], row["target_text"], candidates)
    except Exception:
        choice = 0
    if 1 <= choice <= len(candidates):
        sid, name = candidates[choice - 1]
        _set(row["id"], "resolved", sid, name, method)
        return True
    _set(row["id"], "unresolved", None, None, method)
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
    _ensure_columns()
    print("Loading speaker index ...")
    by_parl, _ = B._load_speaker_index()

    # Person targets not yet resolved (interjections already have their target set).
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, target_text, country, source_dataset, date, context, sentence "
            "FROM accusations "
            "WHERE target_status='success' AND target_type='person' "
            "  AND is_interjection=0 AND resolve_status='pending'"
            + (f" LIMIT {args.limit}" if args.limit else "")
        ).fetchall()]
    print(f"{len(rows):,} person targets to resolve")

    # ---- Pass 1: deterministic ----
    stats = {"exact": 0, "unresolved": 0, "no_surname": 0}
    llm_jobs = []   # (row, candidates, method)
    for row in tqdm(rows, desc="Deterministic"):
        surname = B._surname(row["target_text"] or "")
        if not surname:
            _set(row["id"], "no_surname"); stats["no_surname"] += 1; continue
        kind, cands = _candidates(surname, row["country"], row["source_dataset"],
                                  row["date"], by_parl)
        if kind == "exact" and len(cands) == 1:
            _set(row["id"], "resolved", cands[0][0], cands[0][1], "exact")
            stats["exact"] += 1
        elif kind == "exact":
            llm_jobs.append((row, cands, "llm_ambiguous"))
        elif kind == "fuzzy":
            llm_jobs.append((row, cands, "llm_fuzzy"))
        else:
            _set(row["id"], "unresolved"); stats["unresolved"] += 1

    print(f"\nDeterministic: exact={stats['exact']:,}  "
          f"unresolved={stats['unresolved']:,}  no_surname={stats['no_surname']:,}")
    print(f"LLM tail: {len(llm_jobs):,} (ambiguous + fuzzy)")

    if not llm_jobs:
        return

    # ---- Pass 2: LLM disambiguation/confirmation on the tail ----
    _loaded = None
    n_ok = 0
    try:
        acquire_llm_lock("resolve_targets", config.MODEL)
        print(f"Loading model {config.MODEL} ...")
        _loaded = load_model(config.MODEL, context_length=config.LLM_CONTEXT_LENGTH,
                             num_parallel=config.LLM_NUM_PARALLEL).get("instance_id")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_resolve_one_llm, r, c, m): 1 for (r, c, m) in llm_jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="LLM tail"):
                n_ok += bool(fut.result())
    finally:
        if _loaded:
            try:
                unload_model(_loaded)
            except Exception:
                pass
        release_llm_lock()

    print(f"\nDone.  LLM-resolved {n_ok:,} of {len(llm_jobs):,} tail cases.")
    with get_conn() as conn:
        for st, n in conn.execute(
            "SELECT resolve_status, COUNT(*) FROM accusations "
            "WHERE target_type='person' AND is_interjection=0 GROUP BY resolve_status"):
            print(f"  {st}: {n:,}")


if __name__ == "__main__":
    main()
