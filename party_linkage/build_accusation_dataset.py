"""
build_accusation_dataset.py
===========================
Analysis-ready ACCUSATION dataset: one row per accusation (532k) with the full
accuser variable set and, when the target is a resolved speaker, the full target
variable set too (individual-level + party-level).

Output: parquet at config.ACCUSATION_PARQUET

Usage:
    python3 build_accusation_dataset.py [--limit N]
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

import config
from engine import Engine, VAR_KEYS

# Columns carried straight from the accusations table.
ACC_COLS = [
    "id", "source_dataset", "source_dataset_type", "source_file",
    "source_speech_id", "sentence_idx", "date", "country",
    "sentence", "context", "lie_score",
    "speaker",  # recorded corpus speaker (host for interjections)
    "is_interjection", "interjector_raw", "accuser_speaker", "accuser_name", "accuser_match",
    "target_type", "target_text", "resolve_status", "resolved_speaker_id", "resolved_name",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("Building engine ...")
    eng = Engine()

    con = sqlite3.connect(config.ACC_DB)
    con.row_factory = sqlite3.Row
    have = {r[1] for r in con.execute("PRAGMA table_info(accusations)").fetchall()}
    sel = [c for c in ACC_COLS if c in have]
    missing = [c for c in ACC_COLS if c not in have]
    if missing:
        print(f"  note: absent columns filled null -> {missing}")
    rows = con.execute(
        f"SELECT {','.join(sel)} FROM accusations"
        + (f" LIMIT {args.limit}" if args.limit else "")
    ).fetchall()
    con.close()
    print(f"{len(rows):,} accusations")

    records = []
    for r in rows:
        rec = {c: (r[c] if c in sel else None) for c in ACC_COLS}
        country, dataset, date = rec["country"], rec["source_dataset"], rec["date"]

        # accuser: resolved interjector for interjections, else the recorded speaker
        accuser = (rec["accuser_speaker"] if (rec["is_interjection"] and rec["accuser_speaker"])
                   else rec["speaker"])
        av = eng.vars_for(accuser, country, dataset, date) if accuser else {k: None for k in VAR_KEYS}
        for k in VAR_KEYS:
            rec[f"accuser_{k}"] = av.get(k)

        # target: only when it resolved to a known speaker
        tv = {k: None for k in VAR_KEYS}
        if rec["resolved_speaker_id"]:
            tv = eng.vars_for(rec["resolved_speaker_id"], country, dataset, date)
        for k in VAR_KEYS:
            rec[f"target_{k}"] = tv.get(k)

        records.append(rec)

    df = pd.DataFrame.from_records(records)
    Path(config.ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.ACCUSATION_PARQUET, index=False)
    print(f"\nWrote {len(df):,} rows x {len(df.columns)} cols -> {config.ACCUSATION_PARQUET}")
    print("accuser_match:", df["accuser_match"].value_counts().to_dict())
    print("target rows with speaker vars:",
          int(df["target_match"].notna().sum()))


if __name__ == "__main__":
    main()
