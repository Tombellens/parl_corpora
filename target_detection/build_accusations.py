"""
build_accusations.py
====================
One-time extraction: stream the lielines-scored sentence corpus
(sentence_corpus_predicted.csv, ~149M rows) and pull every detected accusation
(lie_label == LABEL_1) together with a +/- CONTEXT_WINDOW sentence context from
the same speech into the accusations table.

The corpus is written speech-by-speech (sentences contiguous within a speech),
so we buffer the current speech, and when the speech id changes we emit its
accusations with their surrounding context. The accusation sentence is marked
with ">>>" inside the stored context so the LLM knows which sentence to judge.

Usage:
    python3 build_accusations.py [--rebuild] [--progress-every N]
"""

import argparse
import csv
import sys

import config
from db import get_conn, init_db, now_iso

# The corpus has very long speeches; allow big CSV fields.
csv.field_size_limit(sys.maxsize)

SPEECH_KEYS = ("source_dataset", "source_file", "source_speech_id")


def _speech_key(row: dict) -> tuple:
    return tuple(row.get(k) for k in SPEECH_KEYS)


def _make_context(rows: list[dict], i: int, window: int) -> str:
    """Join the +/-window sentences around row i, marking the accusation."""
    lo = max(0, i - window)
    hi = min(len(rows), i + window + 1)
    parts = []
    for j in range(lo, hi):
        text = (rows[j].get("sentence") or "").strip()
        if j == i:
            parts.append(f">>> {text}")
        else:
            parts.append(text)
    return "\n".join(parts)


def _flush_speech(conn, rows: list[dict], batch: list[tuple]) -> int:
    """Find accusations in one speech's rows, append their DB tuples to batch.
    Returns the number of accusations found."""
    n = 0
    for i, row in enumerate(rows):
        if row.get("lie_label") != config.ACCUSATION_LABEL:
            continue
        context = _make_context(rows, i, config.CONTEXT_WINDOW)
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
            row.get("source_file"), row.get("source_speech_id"), idx,
            row.get("date"), row.get("speaker"), row.get("country"),
            (row.get("sentence") or "").strip(), context, score,
        ))
        n += 1
    return n


_INSERT = """
INSERT INTO accusations
  (source_dataset, source_dataset_type, source_file, source_speech_id,
   sentence_idx, date, speaker, country, sentence, context, lie_score)
VALUES (?,?,?,?,?,?,?,?,?,?,?)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Clear the accusations table before extracting")
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

    n_rows = n_acc = n_speeches = 0
    cur_key = None
    speech_rows: list[dict] = []
    batch: list[tuple] = []

    print(f"Streaming {config.PREDICTED_CSV} ...")
    with open(config.PREDICTED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            key = _speech_key(row)
            if key != cur_key and speech_rows:
                n_acc += _flush_speech(None, speech_rows, batch)
                n_speeches += 1
                speech_rows = []
                if len(batch) >= 5000:
                    with get_conn() as conn:
                        conn.executemany(_INSERT, batch)
                    batch = []
            cur_key = key
            speech_rows.append(row)

            if n_rows % args.progress_every == 0:
                print(f"  {n_rows:,} rows | {n_acc:,} accusations | "
                      f"{n_speeches:,} speeches", flush=True)

    # last speech + remaining batch
    if speech_rows:
        n_acc += _flush_speech(None, speech_rows, batch)
        n_speeches += 1
    if batch:
        with get_conn() as conn:
            conn.executemany(_INSERT, batch)

    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM accusations").fetchone()[0]
    print(f"\nDone.  {n_rows:,} rows read | {n_speeches:,} speeches | "
          f"{n_acc:,} accusations extracted | {total:,} rows in table.")


if __name__ == "__main__":
    main()
