"""
build_full_corpus.py
====================
Analysis-ready FULL-CORPUS dataset: one row per sentence (~149M) with the
speaking MP's individual-level + party-level variables attached.

Strategy — this is a JOIN, not a row-by-row transform:

  1. COPY the DISTINCT (speaker, country, source_dataset, date) keys out of the
     CSV with DuckDB.                                    [C++ scan]
  2. Run the variable engine over those keys only — a few million, not 149M —
     streaming the result to a parquet.                  [Python, bounded]
  3. LEFT JOIN that small table back onto the CSV and write the output parquet.
                                                          [C++ join]

Exact-date precision is preserved (keys include the date), and memory stays flat
because every stage streams.

Output: parquet at config.FULL_CORPUS_PARQUET

Usage:
    python3 build_full_corpus.py [--limit N] [--chunk 250000] [--keep-temp]

Requires duckdb:  pip install duckdb
"""

import argparse
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

import config
from engine import Engine, VAR_KEYS

# Canonical name -> candidate column names in the CSV.
COL_ALIASES = {
    "speaker":        ["speaker", "speaker_id", "speaker_name"],
    "country":        ["country"],
    "source_dataset": ["source_dataset", "dataset"],
    "date":           ["date", "sitting_date"],
}

# Engine var keys stored as float64; the rest as strings.
NUM_VARS = {
    "birth_year", "age", "partyfacts_id",
    "vote_share_last", "vote_share_avg", "vote_share_avg_last3", "vote_share_delta",
    "in_cabinet", "is_pm_party", "years_since_government",
    "left_right", "populism", "anti_elitism", "people_centrism",
    "cultural_conservatism", "anti_pluralism",
}


def _fnum(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _resolve_cols(header):
    lower = {c.lower(): c for c in header}
    out = {}
    for canon, aliases in COL_ALIASES.items():
        col = next((lower[a] for a in aliases if a in lower), None)
        if col is None:
            raise SystemExit(f"CSV missing a column for '{canon}' "
                             f"(tried {aliases}); header={list(header)}")
        out[canon] = col
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only read the first N CSV rows (smoke test)")
    ap.add_argument("--chunk", type=int, default=250_000,
                    help="keys processed per batch through the engine")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    analysis = Path(config.ANALYSIS_DIR)
    analysis.mkdir(parents=True, exist_ok=True)
    keys_pq = analysis / "_tmp_corpus_keys.parquet"
    vars_pq = analysis / "_tmp_speaker_vars.parquet"

    con = duckdb.connect()
    src = (f"read_csv('{config.PREDICTED_CSV}', all_varchar=true, header=true"
           + (f", nrows={args.limit}" if args.limit else "") + ")")

    header = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    cols = _resolve_cols(header)
    print(f"column map: {cols}")
    print(f"csv columns carried through: {len(header)}")

    # ---- 1. distinct keys ---------------------------------------------------
    t0 = time.time()
    print("\n[1/3] extracting distinct (speaker, country, dataset, date) keys ...")
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                   "{cols['speaker']}"        AS k_speaker,
                   "{cols['country']}"        AS k_country,
                   "{cols['source_dataset']}" AS k_dataset,
                   "{cols['date']}"           AS k_date
            FROM {src}
            WHERE "{cols['speaker']}" IS NOT NULL
        ) TO '{keys_pq}' (FORMAT PARQUET)
    """)
    n_keys = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{keys_pq}')").fetchone()[0]
    print(f"      {n_keys:,} distinct keys  ({time.time()-t0:.0f}s)")

    # ---- 2. engine over the keys only --------------------------------------
    print("\n[2/3] building engine ...")
    eng = Engine()

    fields = [("k_speaker", pa.string()), ("k_country", pa.string()),
              ("k_dataset", pa.string()), ("k_date", pa.string())]
    for k in VAR_KEYS:
        fields.append((f"speaker_{k}",
                       pa.float64() if k in NUM_VARS else pa.string()))
    schema = pa.schema(fields)

    t0 = time.time()
    print(f"      resolving variables for {n_keys:,} keys ...")
    writer = pq.ParquetWriter(vars_pq, schema)
    done = 0
    for batch in pq.ParquetFile(keys_pq).iter_batches(batch_size=args.chunk):
        recs = []
        for sp, ct, ds, dt in zip(batch.column("k_speaker").to_pylist(),
                                  batch.column("k_country").to_pylist(),
                                  batch.column("k_dataset").to_pylist(),
                                  batch.column("k_date").to_pylist()):
            v = eng.vars_for(sp, ct, ds, dt)
            row = {"k_speaker": sp, "k_country": ct, "k_dataset": ds, "k_date": dt}
            for k in VAR_KEYS:
                val = v.get(k)
                row[f"speaker_{k}"] = _fnum(val) if k in NUM_VARS else (
                    None if val is None else str(val))
            recs.append(row)
        writer.write_table(pa.Table.from_pylist(recs, schema=schema))
        done += len(recs)
        eng._cache.clear()          # keys are distinct: caching only costs memory
        print(f"      {done:,}/{n_keys:,} keys ...", end="\r", flush=True)
    writer.close()
    print(f"\n      done ({time.time()-t0:.0f}s)")

    # ---- 3. join back onto the corpus --------------------------------------
    t0 = time.time()
    print("\n[3/3] joining onto the corpus and writing parquet ...")
    con.execute(f"""
        COPY (
            SELECT c.*, v.* EXCLUDE (k_speaker, k_country, k_dataset, k_date)
            FROM {src} c
            LEFT JOIN read_parquet('{vars_pq}') v
              ON  c."{cols['speaker']}"        IS NOT DISTINCT FROM v.k_speaker
              AND c."{cols['country']}"        IS NOT DISTINCT FROM v.k_country
              AND c."{cols['source_dataset']}" IS NOT DISTINCT FROM v.k_dataset
              AND c."{cols['date']}"           IS NOT DISTINCT FROM v.k_date
        ) TO '{config.FULL_CORPUS_PARQUET}' (FORMAT PARQUET)
    """)
    n_out = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{config.FULL_CORPUS_PARQUET}')"
    ).fetchone()[0]
    print(f"      {n_out:,} rows written ({time.time()-t0:.0f}s)")

    matched = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{config.FULL_CORPUS_PARQUET}')
        WHERE speaker_match = 'resolved'
    """).fetchone()[0]
    print(f"\nDone. {n_out:,} rows -> {config.FULL_CORPUS_PARQUET}")
    print(f"  speaker_match='resolved': {matched:,} ({matched/max(n_out,1)*100:.1f}%)")

    if not args.keep_temp:
        keys_pq.unlink(missing_ok=True)
        vars_pq.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
