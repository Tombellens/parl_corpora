"""
data.py — read-only loaders for the two analysis-ready parquets.

  load_accusations()   -> pandas.DataFrame (532k rows, small, load whole)
  scan_corpus()        -> polars.LazyFrame (149M rows, push filters/aggs down)
  duck()               -> duckdb connection with both parquets registered as views

The full-corpus parquet is tens of GB — NEVER pd.read_parquet it. Use scan_corpus()
(polars lazy) or duck() (SQL) and only .collect()/.df() the small result.

Paths default to /home/tom/data/analysis; override with env ANALYSIS_DIR.
"""

import os
from pathlib import Path

from .codebooks import EXCLUDED_COUNTRIES

ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", "/home/tom/data/analysis"))
ACCUSATIONS_PARQUET = ANALYSIS_DIR / "accusations_dataset.parquet"
FULL_CORPUS_PARQUET = ANALYSIS_DIR / "full_corpus_dataset.parquet"

# lielines threshold: LABEL_1 (accusation) probability above which a sentence counts
# as an accusation in the full corpus. Adjust once, everywhere.
LIE_THRESHOLD = 0.5

# SQL literal list of excluded ISO2 codes, e.g. "'IS','BA','GR','LV'"
_EXCL_SQL = ",".join(f"'{c}'" for c in sorted(EXCLUDED_COUNTRIES))


def load_accusations(exclude_countries=True, columns=None):
    """The 532k accusation dataset as a pandas DataFrame.

    By default drops the excluded countries (codebooks.EXCLUDED_COUNTRIES);
    pass exclude_countries=False to get the raw set. `columns` restricts the
    read (the parquet carries sentence + context text, so subsetting matters).
    """
    import pandas as pd
    if columns is not None:
        columns = list(columns)
        if "country" not in columns:
            columns.append("country")       # needed for the filter
    df = pd.read_parquet(ACCUSATIONS_PARQUET, columns=columns)
    if exclude_countries:
        df = df[~df["country"].isin(EXCLUDED_COUNTRIES)].reset_index(drop=True)
    return df


def scan_corpus(exclude_countries=True):
    """The 149M-row full corpus as a polars LazyFrame (nothing read until collect).

    Excluded countries are filtered lazily by default (pushed down to the scan).
    """
    import polars as pl
    lf = pl.scan_parquet(FULL_CORPUS_PARQUET)
    if exclude_countries:
        lf = lf.filter(~pl.col("country").is_in(list(EXCLUDED_COUNTRIES)))
    return lf


def duck(exclude_countries=True):
    """DuckDB connection with `accusations` and `corpus` views over the parquets.

    By default the views already exclude EXCLUDED_COUNTRIES, so every query is
    filtered. Pass exclude_countries=False for raw views.
    """
    import duckdb
    con = duckdb.connect()
    where = f" WHERE country NOT IN ({_EXCL_SQL})" if exclude_countries else ""
    con.execute(
        f"CREATE VIEW accusations AS SELECT * FROM "
        f"read_parquet('{ACCUSATIONS_PARQUET}'){where}")
    con.execute(
        f"CREATE VIEW corpus AS SELECT * FROM "
        f"read_parquet('{FULL_CORPUS_PARQUET}'){where}")
    return con


def resolved_accusers(df):
    """Filter an accusations DataFrame to rows whose accuser resolved to a party."""
    return df[df["accuser_match"] == "resolved"]


def resolved_targets(df):
    """Filter to accusations whose target resolved to a known speaker (has vars)."""
    return df[df["target_match"] == "resolved"]
