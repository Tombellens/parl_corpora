"""
import_speakers.py
=================
One-time script: reads speaker_names.csv, filters to is_person=True rows,
and inserts them into the speakers table of speaker_enrichment.db.

Safe to re-run: uses INSERT OR IGNORE so existing rows are untouched.

Usage:
    python3 import_speakers.py
"""

import hashlib
import pandas as pd

from config import SPEAKER_NAMES_FILE
from db import init_db, get_conn, now_iso


def make_speaker_id(name_cleaned: str, country: str, source_dataset: str) -> str:
    """Stable, unique ID derived from the three identifying columns."""
    key = f"{name_cleaned}|{country}|{source_dataset}"
    return hashlib.sha1(key.encode()).hexdigest()


def main():
    init_db()

    print(f"Reading {SPEAKER_NAMES_FILE}...")
    df = pd.read_csv(SPEAKER_NAMES_FILE, dtype=str)
    print(f"  {len(df):,} total rows")

    # Keep only confirmed persons
    persons = df[df["is_person"].str.lower() == "true"].copy()
    print(f"  {len(persons):,} rows with is_person=True")

    persons["speaker_id"] = persons.apply(
        lambda r: make_speaker_id(
            str(r.get("name_cleaned", "") or ""),
            str(r.get("country",       "") or ""),
            str(r.get("source_dataset","") or ""),
        ),
        axis=1,
    )

    # Check for duplicates (same id from different rows)
    dup = persons.duplicated("speaker_id", keep=False)
    if dup.any():
        print(f"  Warning: {dup.sum()} rows share a speaker_id — only first kept.")
        persons = persons.drop_duplicates("speaker_id")

    now = now_iso()
    rows = [
        (
            row["speaker_id"],
            row.get("name_cleaned")    or None,
            row.get("country")         or None,
            row.get("source_dataset")  or None,
            row.get("source_dataset_type") or None,
            row.get("min_date")        or None,
            row.get("max_date")        or None,
            int(row["n_sentences"]) if str(row.get("n_sentences","")).isdigit() else None,
            now,
            now,
        )
        for _, row in persons.iterrows()
    ]

    with get_conn() as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO speakers
               (speaker_id, name_cleaned, country, source_dataset, source_dataset_type,
                min_date, max_date, n_sentences, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        total = conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]

    print(f"  Inserted (or skipped if existing): {len(rows):,} rows")
    print(f"  Total rows in speakers table: {total:,}")
    print("Done.")


if __name__ == "__main__":
    main()
