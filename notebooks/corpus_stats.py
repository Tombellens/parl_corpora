"""
corpus_stats.py

Prints a table of sentence counts, min year, and max year
per source_dataset × country, reading sentence_corpus.csv in chunks.
"""

import pandas as pd

INPUT_FILE = "/home/tom/data/sentence_corpus.csv"
CHUNK_SIZE = 1_000_000

agg = {}  # (source_dataset, country) -> {"count": int, "min_year": int, "max_year": int}

print(f"Reading {INPUT_FILE} ...")

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["source_dataset", "country", "date"],
    chunksize=CHUNK_SIZE,
    low_memory=False,
):
    chunk["year"] = pd.to_datetime(chunk["date"], errors="coerce").dt.year

    for (dataset, country), grp in chunk.groupby(["source_dataset", "country"]):
        key = (dataset, country)
        years = grp["year"].dropna()
        entry = agg.setdefault(key, {"count": 0, "min_year": None, "max_year": None})
        entry["count"] += len(grp)
        if len(years):
            mn = int(years.min())
            mx = int(years.max())
            entry["min_year"] = mn if entry["min_year"] is None else min(entry["min_year"], mn)
            entry["max_year"] = mx if entry["max_year"] is None else max(entry["max_year"], mx)

# Build DataFrame
rows = [
    {"dataset": k[0], "country": k[1],
     "sentences": v["count"],
     "min_year": v["min_year"],
     "max_year": v["max_year"]}
    for k, v in agg.items()
]
df = (
    pd.DataFrame(rows)
    .sort_values(["dataset", "country"])
    .reset_index(drop=True)
)

df["sentences"] = df["sentences"].apply(lambda x: f"{x:,}")

# Print
col_w = {
    "dataset":   max(df["dataset"].str.len().max(), 7),
    "country":   max(df["country"].str.len().max(), 7),
    "sentences": max(df["sentences"].str.len().max(), 9),
    "min_year":  8,
    "max_year":  8,
}

header = (f"{'Dataset':<{col_w['dataset']}}  "
          f"{'Country':<{col_w['country']}}  "
          f"{'Sentences':>{col_w['sentences']}}  "
          f"{'Min year':>{col_w['min_year']}}  "
          f"{'Max year':>{col_w['max_year']}}")
sep = "-" * len(header)

print(f"\n{sep}\n{header}\n{sep}")
for _, row in df.iterrows():
    print(f"{row['dataset']:<{col_w['dataset']}}  "
          f"{row['country']:<{col_w['country']}}  "
          f"{row['sentences']:>{col_w['sentences']}}  "
          f"{str(row['min_year'] or ''):>{col_w['min_year']}}  "
          f"{str(row['max_year'] or ''):>{col_w['max_year']}}")
print(sep)

# Totals
total = sum(int(r["sentences"].replace(",", "")) for _, r in df.iterrows())
print(f"\nTotal sentences: {total:,}")

# Also save as CSV
out_csv = INPUT_FILE.replace("sentence_corpus.csv", "corpus_stats.csv")
df.to_csv(out_csv, index=False)
print(f"Saved to {out_csv}")
