"""
build_accusations.py
====================
One-time extraction: stream the lielines-scored sentence corpus
(sentence_corpus_predicted.csv, ~149M rows) and pull every detected accusation
(lie_label == LABEL_1) with a speaker-labelled CONTEXT window into the
accusations table.

Context design: the PREV_CONTEXT_SENTENCES sentences immediately preceding the
accusation in corpus order — crossing speaker/turn boundaries — plus the
accusation itself (marked ">>>"). Each line is grouped under its speaker's full
name (PAD ids resolved via speaker_names.csv), so the LLM can see turn structure
and who spoke just before the accusation (often the target). The rolling window
is reset at source_file boundaries so context never bleeds across transcripts.

Usage:
    python3 build_accusations.py [--rebuild] [--progress-every N]
"""

import argparse
import csv
import sys
from collections import deque

import pandas as pd

import config
from db import get_conn, init_db

csv.field_size_limit(sys.maxsize)


def _load_pad2name() -> dict:
    """corpus speaker string -> resolved full name (name_cleaned)."""
    df = pd.read_csv(config.SPEAKER_NAMES, dtype=str).fillna("")
    out = {}
    for r in df.itertuples(index=False):
        out[r.speaker] = getattr(r, "name_cleaned", "") or r.speaker
    return out


def _resolve(speaker: str, pad2name: dict) -> str:
    name = pad2name.get(speaker or "", "")
    return name or (speaker or "unknown")


def _build_context(prev, acc_speaker, acc_sentence, pad2name) -> str:
    """prev: iterable of (speaker, sentence) preceding the accusation."""
    lines = []
    last = object()
    for spk, sent in prev:
        if spk != last:
            lines.append(f"[{_resolve(spk, pad2name)}]")
            last = spk
        lines.append((sent or "").strip())
    if acc_speaker != last:
        lines.append(f"[{_resolve(acc_speaker, pad2name)}]")
    lines.append(f">>> {(acc_sentence or '').strip()}")
    return "\n".join(lines)


_INSERT = """
INSERT INTO accusations
  (source_dataset, source_dataset_type, source_file, source_speech_id,
   sentence_idx, date, speaker, country, sentence, context, lie_score)
VALUES (?,?,?,?,?,?,?,?,?,?,?)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--progress-every", type=int, default=5_000_000)
    args = ap.parse_args()

    init_db()
    with get_conn() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM accusations").fetchone()[0]
        if existing and not args.rebuild:
            print(f"accusations table already has {existing:,} rows. "
                  f"Use --rebuild to start over. Exiting.")
            return
        if args.rebuild:
            conn.execute("DELETE FROM accusations")
            print("Cleared existing accusations.")

    print("Loading speaker name index ...")
    pad2name = _load_pad2name()

    n_rows = n_acc = 0
    cur_file = None
    prev = deque(maxlen=config.PREV_CONTEXT_SENTENCES)
    batch = []

    print(f"Streaming {config.PREDICTED_CSV} ...")
    with open(config.PREDICTED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            src_file = row.get("source_file")
            if src_file != cur_file:
                prev.clear()                 # don't cross transcripts
                cur_file = src_file

            if row.get("lie_label") == config.ACCUSATION_LABEL:
                context = _build_context(prev, row.get("speaker"),
                                         row.get("sentence"), pad2name)
                try:
                    idx = int(row.get("sentence_idx"))
                except (TypeError, ValueError):
                    idx = None
                try:
                    score = float(row.get("lie_score"))
                except (TypeError, ValueError):
                    score = None
                batch.append((
                    row.get("source_dataset"), row.get("source_dataset_type"),
                    src_file, row.get("source_speech_id"), idx,
                    row.get("date"), row.get("speaker"), row.get("country"),
                    (row.get("sentence") or "").strip(), context, score,
                ))
                n_acc += 1
                if len(batch) >= 5000:
                    with get_conn() as conn:
                        conn.executemany(_INSERT, batch)
                    batch = []

            prev.append((row.get("speaker"), row.get("sentence")))

            if n_rows % args.progress_every == 0:
                print(f"  {n_rows:,} rows | {n_acc:,} accusations", flush=True)

    if batch:
        with get_conn() as conn:
            conn.executemany(_INSERT, batch)

    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM accusations").fetchone()[0]
    print(f"\nDone.  {n_rows:,} rows read | {n_acc:,} accusations extracted | "
          f"{total:,} rows in table.")


if __name__ == "__main__":
    main()
