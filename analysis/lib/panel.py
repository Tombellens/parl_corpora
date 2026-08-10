"""
panel.py — build & load the SPEAKER-YEAR PANEL, the workhorse table for H2–H4.

One row per (speaker_id, country, source_dataset, year):
  n_sentences              exposure: sentences spoken that year (full corpus)
  n_accusations_made       accusations made (accuser side, incl. interjections)
  n_accusations_received   accusations received (resolved targets only!)
  + individual covariates  gender, birth_year -> age, highest_isced, career_sectors
  + party covariates       partyfacts_id, party_name, left_right, populism,
                           anti_elitism, people_centrism, in_cabinet, is_pm_party,
                           years_since_government, vote_share_last

All accuser/accusee-side models are then count models on this table with
offset(log(n_sentences)) and country/dataset/year fixed effects.

CAVEAT: n_accusations_received only counts accusations whose target RESOLVED to
a known speaker (~50%, varying strongly by dataset). Accusee-side models must
include dataset FE and should be robustness-checked on high-resolution datasets.

Build once (slow: aggregates 149M rows), cached as parquet:
    python3 -m lib.panel            # from the analysis/ folder
    python3 -m lib.panel --force    # rebuild
"""

from . import data

PANEL_PARQUET = data.ANALYSIS_DIR / "speaker_year_panel.parquet"

_BUILD_SQL = """
COPY (
WITH corpus_agg AS (
    SELECT
        speaker_speaker_id AS speaker_id,
        country, source_dataset,
        CAST(substr(date, 1, 4) AS INT)          AS year,
        COUNT(*)                                 AS n_sentences,
        any_value(speaker_gender)                AS gender,
        any_value(speaker_birth_year)            AS birth_year,
        any_value(speaker_highest_isced)         AS highest_isced,
        any_value(speaker_career_sectors)        AS career_sectors,
        any_value(speaker_partyfacts_id)         AS partyfacts_id,
        any_value(speaker_party_name)            AS party_name,
        any_value(speaker_left_right)            AS left_right,
        any_value(speaker_populism)              AS populism,
        any_value(speaker_anti_elitism)          AS anti_elitism,
        any_value(speaker_people_centrism)       AS people_centrism,
        any_value(speaker_in_cabinet)            AS in_cabinet,
        any_value(speaker_is_pm_party)           AS is_pm_party,
        any_value(speaker_years_since_government) AS years_since_government,
        any_value(speaker_vote_share_last)       AS vote_share_last,
        any_value(speaker_match)                 AS speaker_match
    FROM corpus
    WHERE speaker_speaker_id IS NOT NULL
      AND date IS NOT NULL AND length(date) >= 4
    GROUP BY 1, 2, 3, 4
),
made AS (
    SELECT accuser_speaker_id AS speaker_id, country, source_dataset,
           CAST(substr(date, 1, 4) AS INT) AS year,
           COUNT(*) AS n_accusations_made
    FROM accusations
    WHERE accuser_speaker_id IS NOT NULL
    GROUP BY 1, 2, 3, 4
),
received AS (
    SELECT target_speaker_id AS speaker_id, country, source_dataset,
           CAST(substr(date, 1, 4) AS INT) AS year,
           COUNT(*) AS n_accusations_received
    FROM accusations
    WHERE target_speaker_id IS NOT NULL
    GROUP BY 1, 2, 3, 4
)
SELECT c.*,
       COALESCE(m.n_accusations_made, 0)     AS n_accusations_made,
       COALESCE(r.n_accusations_received, 0) AS n_accusations_received
FROM corpus_agg c
LEFT JOIN made     m USING (speaker_id, country, source_dataset, year)
LEFT JOIN received r USING (speaker_id, country, source_dataset, year)
) TO '{out}' (FORMAT PARQUET)
"""


def build_panel(force=False):
    """Aggregate both parquets into the speaker-year panel (cached)."""
    if PANEL_PARQUET.exists() and not force:
        print(f"panel exists: {PANEL_PARQUET} (use --force to rebuild)")
        return
    con = data.duck()                       # views already exclude countries
    print("aggregating 149M corpus rows + accusations ... (several minutes)")
    con.execute(_BUILD_SQL.format(out=PANEL_PARQUET))
    n = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{PANEL_PARQUET}')").fetchone()[0]
    print(f"wrote {n:,} speaker-year rows -> {PANEL_PARQUET}")


def load_panel():
    """The panel as pandas, with derived columns ready for modelling."""
    import numpy as np
    import pandas as pd
    df = pd.read_parquet(PANEL_PARQUET)
    df["age"] = df["year"] - df["birth_year"]
    df["highest_isced"] = pd.to_numeric(df["highest_isced"], errors="coerce")
    df["female"] = (df["gender"].str.lower() == "female").astype("Int64")
    df.loc[df["gender"].isna(), "female"] = pd.NA
    df["log_exposure"] = np.log(df["n_sentences"])
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    build_panel(force=ap.parse_args().force)
