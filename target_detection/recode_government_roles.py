"""
recode_government_roles.py
=========================
Deterministic taxonomy correction: a government function holder (minister,
chancellor, state secretary, ...) belongs to target_type 'government', not
'person'. The target detector sometimes coded such role-mentions as 'person'
(e.g. "the Finance Minister", "Minister of the Interior Caspar", "Mr
Finanzminister"). This recodes those person rows to 'government', keying on the
MENTION text — a plain name ("Dr. Martin Graf") stays 'person'.

Recoded rows no longer need person resolution, so their resolve_status is set to
'not_applicable'.

Usage:
    python3 recode_government_roles.py [--dry-run]
"""

import argparse
import re

from db import get_conn, init_db, now_iso

# Government-executive role terms (English-translated corpus, with some German
# compounds that survive translation). Substring match, case-insensitive.
# "minister" as a substring also catches "Finanzminister"/"Bundesminister"/
# "Finance Minister"/"the Minister". Verbs like "administer" don't appear as
# accusation targets, so the substring is safe in practice.
GOVROLE_RE = re.compile(
    r"(minister|chancellor|kanzler|staatssekret|secretary of state|state secretary)",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, target_text FROM accusations "
            "WHERE target_status='success' AND target_type='person'"
        ).fetchall()

    hits = [r["id"] for r in rows if GOVROLE_RE.search(r["target_text"] or "")]
    sample = [r["target_text"] for r in rows if GOVROLE_RE.search(r["target_text"] or "")][:20]

    print(f"person targets: {len(rows):,}")
    print(f"  -> recode to government (minister-like role): {len(hits):,}")
    print("sample mentions:")
    for s in sample:
        print("  ", repr(s))

    if args.dry_run:
        print("\n--dry-run: no writes.")
        return

    ts = now_iso()
    with get_conn() as conn:
        conn.executemany(
            "UPDATE accusations SET target_type='government', "
            "resolve_status='not_applicable', "
            "target_error=COALESCE(target_error,'')||' [recoded person->government: gov role]', "
            "resolved_speaker_id=NULL, resolved_name=NULL, resolve_method=NULL "
            "WHERE id=?",
            [(i,) for i in hits],
        )
    print(f"\nRecoded {len(hits):,} rows person -> government.")


if __name__ == "__main__":
    main()
