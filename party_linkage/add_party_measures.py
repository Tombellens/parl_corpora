"""
add_party_measures.py
=====================
Add newly-introduced V-Party measures to an EXISTING dataset parquet, without
re-deriving anything.

This works because the party-level measures are a pure function of
(partyfacts_id, date), and partyfacts_id is already stored in both datasets.
So we only need:

  1. the distinct (partyfacts_id, date) pairs present in the file,
  2. the new measures looked up for those pairs,
  3. a LEFT JOIN writing a new parquet with the extra columns.

Nothing about speakers, parties, ParlGov or the corpus CSV is recomputed.

Usage:
    # full corpus (speaker_* prefix)
    python3 add_party_measures.py --dataset corpus

    # accusation dataset (accuser_* and target_* prefixes)
    python3 add_party_measures.py --dataset accusations

    # swap the new file into place when done
    python3 add_party_measures.py --dataset corpus --replace

Requires duckdb:  pip install duckdb
"""

import argparse
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

import config
from build_party_vars import load_vparty, vparty_vars, VPARTY_MEASURES

DATASETS = {
    "corpus":      (config.FULL_CORPUS_PARQUET, ["speaker_"], "date"),
    "accusations": (config.ACCUSATION_PARQUET, ["accuser_", "target_"], "date"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--replace", action="store_true",
                    help="move the new file over the original when finished")
    args = ap.parse_args()

    path, prefixes, date_col = DATASETS[args.dataset]
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    out_path = path.with_suffix(".new.parquet")

    con = duckdb.connect()
    existing = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]

    # which (prefix, measure) columns are missing?
    todo = [(p, m) for p in prefixes for m in VPARTY_MEASURES
            if f"{p}{m}" not in existing]
    if not todo:
        print("nothing to add — all measures already present.")
        return
    print(f"adding {len(todo)} columns: {[p + m for p, m in todo]}")

    # party id column per prefix must exist
    for p in {p for p, _ in todo}:
        if f"{p}partyfacts_id" not in existing:
            raise SystemExit(f"{p}partyfacts_id missing — cannot join on party id")

    print("\nloading V-Party ...")
    vp = load_vparty()

    tmp = path.parent / f"_tmp_{args.dataset}_newvars.parquet"
    added_cols = []

    for prefix in {p for p, _ in todo}:
        measures = [m for p, m in todo if p == prefix]
        pid_col = f"{prefix}partyfacts_id"

        t0 = time.time()
        keys_pq = path.parent / f"_tmp_{args.dataset}_{prefix}keys.parquet"
        con.execute(f"""
            COPY (
                SELECT DISTINCT CAST("{pid_col}" AS BIGINT) AS pfid,
                       "{date_col}" AS d
                FROM read_parquet('{path}')
                WHERE "{pid_col}" IS NOT NULL AND "{date_col}" IS NOT NULL
            ) TO '{keys_pq}' (FORMAT PARQUET)
        """)
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{keys_pq}')").fetchone()[0]
        print(f"\n[{prefix}] {n:,} distinct (party, date) pairs ({time.time()-t0:.0f}s)")

        schema = pa.schema([("pfid", pa.int64()), ("d", pa.string())]
                           + [(m, pa.float64()) for m in measures])
        vars_pq = path.parent / f"_tmp_{args.dataset}_{prefix}vars.parquet"
        writer = pq.ParquetWriter(vars_pq, schema)
        t0 = time.time()
        for batch in pq.ParquetFile(keys_pq).iter_batches(batch_size=args.chunk):
            recs = []
            for pfid, d in zip(batch.column("pfid").to_pylist(),
                               batch.column("d").to_pylist()):
                vv = vparty_vars(pfid, str(d)[:10], vp)
                row = {"pfid": pfid, "d": d}
                for m in measures:
                    val = vv.get(m)
                    row[m] = None if val is None else float(val)
                recs.append(row)
            writer.write_table(pa.Table.from_pylist(recs, schema=schema))
        writer.close()
        print(f"[{prefix}] measures resolved ({time.time()-t0:.0f}s)")

        added_cols.append((prefix, pid_col, measures, vars_pq))
        keys_pq.unlink(missing_ok=True)

    # ---- single join writing the new file ---------------------------------
    t0 = time.time()
    print("\njoining and writing new parquet ...")
    selects, joins = ["c.*"], []
    for i, (prefix, pid_col, measures, vars_pq) in enumerate(added_cols):
        a = f"v{i}"
        selects += [f'{a}."{m}" AS "{prefix}{m}"' for m in measures]
        joins.append(
            f"LEFT JOIN read_parquet('{vars_pq}') {a} "
            f'ON CAST(c."{pid_col}" AS BIGINT) IS NOT DISTINCT FROM {a}.pfid '
            f'AND c."{date_col}" IS NOT DISTINCT FROM {a}.d')
    con.execute(f"""
        COPY (
            SELECT {', '.join(selects)}
            FROM read_parquet('{path}') c
            {' '.join(joins)}
        ) TO '{out_path}' (FORMAT PARQUET)
    """)
    n_out = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    print(f"wrote {n_out:,} rows -> {out_path}  ({time.time()-t0:.0f}s)")

    # coverage of the new columns
    for prefix, _, measures, _ in added_cols:
        for m in measures:
            c = con.execute(f"""SELECT COUNT("{prefix}{m}")
                                FROM read_parquet('{out_path}')""").fetchone()[0]
            print(f"  {prefix}{m}: {c:,} non-null ({c/max(n_out,1)*100:.1f}%)")

    for _, _, _, vars_pq in added_cols:
        vars_pq.unlink(missing_ok=True)
    if tmp.exists():
        tmp.unlink()

    if args.replace:
        out_path.replace(path)
        print(f"\nreplaced {path}")
    else:
        print(f"\nreview it, then swap in with:\n  mv {out_path} {path}")


if __name__ == "__main__":
    main()
