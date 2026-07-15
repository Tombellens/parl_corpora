"""
build_interjections.py
======================
Detect and correctly attribute INTERJECTION accusations.

Some accusations are heckles embedded inline in another speaker's speech,
formatted "Name: <accusation>" (e.g. "Dr. Fekter: That's not true!"). In the
corpus these inherit the HOST speaker's id, so the recorded accuser is wrong —
it is the person being heckled, not the heckler.

For each such accusation we:
  1. flag is_interjection,
  2. parse the interjector name from the prefix,
  3. resolve it to a real speaker in the SAME parliament active in the SAME
     period (country + source_dataset + date overlap + surname match),
  4. take the recorded host speaker as the TARGET (safe structural assumption),
     and mark target_status='success' so the LLM classifier skips these rows.

Non-interjection accusations are left untouched for the LLM target classifier.

Usage:
    python3 build_interjections.py [--limit N] [--dry-run]
"""

import argparse
import re
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

import config
from db import get_conn, init_db, now_iso, SUCCESS

# Prefix like "Dr. Fekter:", "Federal Minister Dr. Fassleven:", "Tichy-Schreder:"
# Require a NAME: every token capitalised (or a known lowercase particle), up to
# 5 tokens. This rejects reported speech / normal sentences that merely contain a
# capitalised word before a colon ("He wrote:", "Call to the liberals:",
# "Mrs Reitsamer said:").
_NAME_TOK = r"[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.'\-]*"
_PARTICLE = r"von|van|de|der|den|zu|di|da"
PREFIX_RE = re.compile(rf"^({_NAME_TOK}(?:\s+(?:{_NAME_TOK}|{_PARTICLE})){{0,4}}):\s")

# Honorifics / titles to strip before taking the surname.
TITLE_RE = re.compile(
    r"^(?:the\s+)?(?:federal\s+)?(?:minister|bundesminister(?:in)?|staatssekretär(?:in)?|"
    r"präsident(?:in)?|abg\.?|mag\.?|dr\.?|dipl\.?-?\s?ing\.?|prof\.?|"
    r"mr|mrs|ms|miss|hon\.?)\s+",
    re.IGNORECASE,
)

NEW_COLUMNS = {
    "is_interjection":  "INTEGER DEFAULT 0",
    "interjector_raw":  "TEXT",
    "accuser_speaker":  "TEXT",    # resolved host-parliament speaker id of the true accuser
    "accuser_name":     "TEXT",    # its name_cleaned
    "accuser_match":    "TEXT",    # 'resolved' | 'ambiguous' | 'unresolved'
}


def _ensure_columns():
    with get_conn() as conn:
        have = {r[1] for r in conn.execute("PRAGMA table_info(accusations)").fetchall()}
        for col, decl in NEW_COLUMNS.items():
            if col not in have:
                conn.execute(f"ALTER TABLE accusations ADD COLUMN {col} {decl}")
                print(f"  added column {col}")


def _surname(name: str) -> str:
    """Strip titles, return the lowercased surname (last whitespace token)."""
    n = name.strip()
    prev = None
    while n != prev:                       # strip stacked titles
        prev = n
        n = TITLE_RE.sub("", n).strip()
    n = n.strip(".,;:").strip()
    if not n:
        return ""
    return n.split()[-1].casefold()


def _load_speaker_index():
    """
    Return:
      by_parl : {(country, source_dataset): [(speaker, name_cleaned, surname, min, max)]}
      pad2name: {speaker_id: name_cleaned}
    """
    df = pd.read_csv(config.SPEAKER_NAMES, dtype=str).fillna("")
    by_parl = defaultdict(list)
    pad2name = {}
    for r in df.itertuples(index=False):
        name = getattr(r, "name_cleaned", "") or getattr(r, "speaker", "")
        pad2name[r.speaker] = name
        by_parl[(r.country, r.source_dataset)].append(
            (r.speaker, name, _surname(name), r.min_date, r.max_date)
        )
    return by_parl, pad2name


FUZZY_THRESHOLD = 0.82   # min surname similarity for a fuzzy accuser match
FUZZY_MARGIN    = 0.05   # best must beat the runner-up by this much


def _resolve_accuser(surname, country, dataset, date, by_parl):
    """Return (speaker_id, name, match) for a surname in the same parliament and
    active period. match in {'resolved','fuzzy','ambiguous','unresolved'}.
    Exact surname match is preferred; a high-confidence fuzzy match recovers
    translation/spelling variants (e.g. 'Applebeck' -> 'Apfelbeck')."""
    cands = by_parl.get((country, dataset), [])
    d = (date or "")[:10]

    def in_period(lo, hi):
        return not (d and lo and hi) or (lo[:10] <= d <= hi[:10])

    exact = [(sid, name) for sid, name, sn, lo, hi in cands
             if sn == surname and in_period(lo, hi)]
    if len(exact) == 1:
        return exact[0][0], exact[0][1], "resolved"
    if len(exact) > 1:
        return None, None, "ambiguous"

    # Fuzzy fallback among date-valid candidates, unique clear best only.
    scored = []
    for sid, name, sn, lo, hi in cands:
        if not sn or not in_period(lo, hi):
            continue
        r = SequenceMatcher(None, surname, sn).ratio()
        if r >= FUZZY_THRESHOLD:
            scored.append((r, sid, name))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= FUZZY_MARGIN:
            return scored[0][1], scored[0][2], "fuzzy"
        return None, None, "ambiguous"
    return None, None, "unresolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Report only, no writes")
    args = ap.parse_args()

    init_db()
    if not args.dry_run:
        _ensure_columns()

    print("Loading speaker index ...")
    by_parl, pad2name = _load_speaker_index()

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, sentence, speaker, country, source_dataset, date FROM accusations"
            + (f" LIMIT {args.limit}" if args.limit else "")
        ).fetchall()

    n_int = 0
    stats = defaultdict(int)
    updates = []
    for row in rows:
        m = PREFIX_RE.match(row["sentence"] or "")
        if not m:
            continue
        n_int += 1
        interjector_raw = m.group(1).strip()
        surname = _surname(interjector_raw)

        host_pad  = row["speaker"]
        host_name = pad2name.get(host_pad, "")
        # If the parsed interjector is actually the host, it is not a misattribution.
        host_surname = _surname(host_name) if host_name else ""

        sid, name, match = _resolve_accuser(
            surname, row["country"], row["source_dataset"], row["date"], by_parl)
        stats[match] += 1
        if surname and surname == host_surname:
            stats["matches_host"] += 1

        updates.append((
            interjector_raw, sid, name, match,           # accuser side
            "person", host_name, host_pad,               # target = host speaker
            row["id"],
        ))

    print(f"\n{n_int:,} interjection accusations found")
    print(f"  accuser resolved (exact) : {stats['resolved']:,}")
    print(f"  accuser resolved (fuzzy) : {stats['fuzzy']:,}")
    print(f"  accuser ambiguous        : {stats['ambiguous']:,}")
    print(f"  accuser unresolved       : {stats['unresolved']:,}")
    print(f"  (parsed == host)         : {stats['matches_host']:,}")

    if args.dry_run:
        print("\n--dry-run: no writes.")
        return

    ts = now_iso()
    with get_conn() as conn:
        conn.executemany(
            """UPDATE accusations SET
                 is_interjection=1, interjector_raw=?, accuser_speaker=?,
                 accuser_name=?, accuser_match=?,
                 target_type=?, target_text=?, resolved_speaker_id=?,
                 target_status='success', resolve_status='success',
                 target_model='structural_interjection', target_prompt_v=?, target_at=?
               WHERE id=?""",
            [(u[0], u[1], u[2], u[3], u[4], u[5], u[6], config.PROMPT_VERSION, ts, u[7])
             for u in updates],
        )
    print(f"\nWrote {len(updates):,} interjection rows "
          f"(target=host speaker, flagged, LLM classifier will skip them).")


if __name__ == "__main__":
    main()
