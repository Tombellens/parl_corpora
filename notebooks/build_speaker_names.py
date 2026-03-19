"""
build_speaker_names.py
======================
Extract all unique speaker names from the base sentence corpus, grouped by
(speaker, country, source_dataset, source_dataset_type), with date range and
sentence count per speaker-dataset combination.

Output: /home/tom/data/speaker_names.csv

Columns:
    speaker              - speaker name string (as in the corpus)
    country              - ISO country code
    source_dataset       - dataset name (e.g. "ParlaMint-FR", "Bundestag", "lipad")
    source_dataset_type  - dataset type (e.g. "parlamint", "parlspeech", "hansard")
    min_date             - earliest date this speaker appears
    max_date             - latest date this speaker appears
    n_sentences          - total number of sentences attributed to this speaker

Usage:
    python3 build_speaker_names.py
    # or in the background:
    nohup python3 build_speaker_names.py > /home/tom/data/speaker_names.log 2>&1 &
"""

import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CORPUS_FILE = "/home/tom/data/sentence_corpus.csv"
OUTPUT_FILE = "/home/tom/data/speaker_names.csv"
CHUNK_SIZE  = 500_000

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(f"Reading corpus: {CORPUS_FILE}")

    GROUP_KEYS = ["speaker", "country", "source_dataset", "source_dataset_type"]
    agg_chunks = []

    for i, chunk in enumerate(pd.read_csv(
        CORPUS_FILE,
        usecols=["speaker", "country", "source_dataset", "source_dataset_type", "date"],
        dtype=str,
        chunksize=CHUNK_SIZE,
    )):
        print(f"  chunk {i:>4}  ({i * CHUNK_SIZE:>12,.0f} rows processed)...")

        chunk["date"] = chunk["date"].fillna("")
        agg = chunk.groupby(GROUP_KEYS, sort=False).agg(
            min_date=("date", "min"),
            max_date=("date", "max"),
            n_sentences=("date", "count"),
        ).reset_index()
        agg_chunks.append(agg)

    print(f"\nMerging {len(agg_chunks)} chunks...")
    combined = pd.concat(agg_chunks, ignore_index=True)
    df = combined.groupby(GROUP_KEYS, sort=False).agg(
        min_date=("min_date", "min"),
        max_date=("max_date", "max"),
        n_sentences=("n_sentences", "sum"),
    ).reset_index()

    print(f"Found {len(df):,} unique speaker-dataset combinations.")
    df = df.sort_values(["country", "source_dataset", "speaker"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    print(df.groupby("source_dataset_type")["speaker"].count().to_string())


if __name__ == "__main__":
    main()
