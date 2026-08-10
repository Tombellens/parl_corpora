"""
country_coverage.py
===================
Which countries make it into the analysis, and where the others drop out.

Three filters remove countries between the raw corpus and a trend estimate:
  1. codebooks.EXCLUDED_COUNTRIES  — excluded on data-quality grounds
  2. MIN_SENTENCES per country-year — sparse years give unstable rates
  3. MIN_YEARS_TREND               — too few years to fit a slope

Run from the analysis/ folder:
    python3 00_overview/country_coverage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lib import data
from lib.codebooks import EXCLUDED_COUNTRIES

MIN_SENTENCES = 10_000
MIN_YEARS_TREND = 10

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 100)


def main():
    # unfiltered view so we can see the excluded countries too
    con = data.duck(exclude_countries=False)

    cy = con.execute("""
        SELECT country,
               CAST(substr(date, 1, 4) AS INT) AS year,
               COUNT(*) AS n_sentences
        FROM corpus
        WHERE date IS NOT NULL AND length(date) >= 4
        GROUP BY 1, 2
    """).df()

    tot = (cy.groupby("country")
             .agg(all_years=("year", "nunique"),
                  sentences=("n_sentences", "sum"),
                  first=("year", "min"), last=("year", "max")))

    dense = cy[cy["n_sentences"] >= MIN_SENTENCES]
    tot["usable_years"] = dense.groupby("country")["year"].nunique()
    tot["usable_years"] = tot["usable_years"].fillna(0).astype(int)

    def verdict(r):
        if r.name in EXCLUDED_COUNTRIES:
            return "EXCLUDED (data quality)"
        if r["usable_years"] == 0:
            return f"no year reaches {MIN_SENTENCES:,} sentences"
        if r["usable_years"] < MIN_YEARS_TREND:
            return f"only {r['usable_years']} usable years (need {MIN_YEARS_TREND})"
        return "in trend analysis"

    tot["status"] = tot.apply(verdict, axis=1)
    tot = tot.sort_values(["status", "sentences"], ascending=[True, False])

    print(f"{len(tot)} countries in the corpus\n")
    print(tot[["first", "last", "all_years", "usable_years",
               "sentences", "status"]].to_string())

    print("\n--- summary ---")
    for status, grp in tot.groupby("status"):
        print(f"{len(grp):>3}  {status}")
        print(f"     {', '.join(sorted(grp.index))}")


if __name__ == "__main__":
    main()
