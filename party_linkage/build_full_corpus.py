"""
build_full_corpus.py
====================
Analysis-ready FULL-CORPUS dataset: one row per sentence (~149M) from the
lielines-scored corpus, with the speaking MP's individual-level + party-level
variables attached and the lie flag retained.

Streams the CSV in chunks and writes a single parquet via pyarrow ParquetWriter
so memory stays flat regardless of corpus size.

Output: parquet at config.FULL_CORPUS_PARQUET

Usage:
    python3 build_full_corpus.py [--limit N] [--chunk 200000]
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from engine import Engine, VAR_KEYS

# Required identity columns (script aborts if absent under any known alias).
COL_ALIASES = {
    "speaker":        ["speaker", "speaker_id", "speaker_name"],
    "country":        ["country"],
    "source_dataset": ["source_dataset", "dataset"],
    "date":           ["date", "sitting_date"],
    "sentence":       ["sentence", "text", "sentence_text"],
    "lie_score":      ["lie_score", "prediction", "pred", "label", "score", "lielines"],
}

# Passed straight through if present.
PASSTHROUGH = ["source_file", "source_speech_id", "sentence_idx",
               "source_dataset_type", "speech_id"]

CACHE_CAP = 4_000_000   # clear engine cache past this many (speaker,date) combos

# engine var keys stored as float64 (rest are strings)
NUM_VARS = {
    "birth_year", "age", "partyfacts_id",
    "vote_share_last", "vote_share_avg", "vote_share_avg_last3", "vote_share_delta",
    "in_cabinet", "is_pm_party", "years_since_government",
    "left_right", "populism", "anti_elitism", "people_centrism",
}


def _fnum(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _fstr(x):
    return None if x is None else str(x)


def _resolve_cols(header):
    """Map canonical name -> actual column in this CSV (or None)."""
    lower = {c.lower(): c for c in header}
    out = {}
    for canon, aliases in COL_ALIASES.items():
        out[canon] = next((lower[a] for a in aliases if a in lower), None)
    for req in ("speaker", "country", "source_dataset", "date"):
        if out[req] is None:
            raise SystemExit(f"CSV missing required column for '{req}' "
                             f"(tried {COL_ALIASES[req]}); header={list(header)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()

    print("Building engine ...")
    eng = Engine()

    header = pd.read_csv(config.PREDICTED_CSV, nrows=0).columns
    cols = _resolve_cols(header)
    passthrough = [c for c in PASSTHROUGH if c in header]
    print(f"  column map: {cols}")
    print(f"  passthrough: {passthrough}")

    # Fixed output schema: raw/passthrough as string, engine vars typed by NUM_VARS.
    base_cols = ["speaker", "country", "source_dataset", "date",
                 "sentence", "lie_score"] + passthrough
    fields = [(c, pa.string()) for c in base_cols]
    for k in VAR_KEYS:
        fields.append((f"speaker_{k}",
                       pa.float64() if k in NUM_VARS else pa.string()))
    schema = pa.schema(fields)

    Path(config.ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(config.FULL_CORPUS_PARQUET, schema)
    total = 0

    reader = pd.read_csv(config.PREDICTED_CSV, chunksize=args.chunk,
                         dtype=str, keep_default_na=False,
                         nrows=args.limit, iterator=True)
    for chunk in reader:
        recs = []
        for r in chunk.itertuples(index=False):
            g = {c: (getattr(r, cols[c]) if cols[c] else None)
                 for c in ("speaker", "country", "source_dataset", "date",
                           "sentence", "lie_score")}
            for p in passthrough:
                g[p] = getattr(r, p)
            v = eng.vars_for(g["speaker"], g["country"],
                             g["source_dataset"], g["date"])
            for k in VAR_KEYS:
                val = v.get(k)
                g[f"speaker_{k}"] = _fnum(val) if k in NUM_VARS else _fstr(val)
            recs.append(g)

        writer.write_table(pa.Table.from_pylist(recs, schema=schema))
        total += len(recs)
        if len(eng._cache) > CACHE_CAP:
            eng._cache.clear()
        print(f"  {total:,} rows written ...", end="\r", flush=True)

    if writer is not None:
        writer.close()
    print(f"\nDone. {total:,} rows -> {config.FULL_CORPUS_PARQUET}")


if __name__ == "__main__":
    main()
