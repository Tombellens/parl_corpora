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

    records = defaultdict(lambda: {
        "min_date":    "9999",
        "max_date":    "0000",
        "n_sentences": 0,
    })

    for i, chunk in enumerate(pd.read_csv(
        CORPUS_FILE,
        usecols=["speaker", "country", "source_dataset", "source_dataset_type", "date"],
        dtype=str,
        chunksize=CHUNK_SIZE,
    )):
        if i % 10 == 0:
            print(f"  chunk {i:>4}  ({i * CHUNK_SIZE:>12,.0f} rows processed)...")

        for _, row in chunk.iterrows():
            key = (
                str(row["speaker"]),
                str(row["country"]),
                str(row["source_dataset"]),
                str(row["source_dataset_type"]),
            )
            d = str(row["date"]) if pd.notna(row["date"]) else ""
            r = records[key]
            r["n_sentences"] += 1
            if d and d < r["min_date"]:
                r["min_date"] = d
            if d and d > r["max_date"]:
                r["max_date"] = d

    print(f"\nFound {len(records):,} unique speaker-dataset combinations.")

    rows = [
        {
            "speaker":             k[0],
            "country":             k[1],
            "source_dataset":      k[2],
            "source_dataset_type": k[3],
            **v,
        }
        for k, v in records.items()
    ]

    df = pd.DataFrame(rows).sort_values(["country", "source_dataset", "speaker"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    print(df.groupby("source_dataset_type")["speaker"].count().to_string())


if __name__ == "__main__":
    main()
