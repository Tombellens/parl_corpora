"""
panel.py — build & load the SPEAKER-YEAR PANEL, the workhorse table for H2–H4.

One row per (speaker_id, country, source_dataset, year):
  n_sentences              exposure: sentences spoken that year (full corpus)
  n_accusations_made       accusations made (accuser side, incl. interjections)
  n_accusations_received   accusations received (resolved targets only!)
  + individual covariates  gender, birth_year -> age, highest_isced, career_sectors
  + party covariates       partyfacts_id, party_name, left_right (ECONOMIC),
                           cultural_conservatism, populism, anti_elitism,
                           people_centrism, anti_pluralism, in_cabinet,
                           is_pm_party, years_since_government, vote_share_last

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
        any_value(speaker_cultural_conservatism) AS cultural_conservatism,
        any_value(speaker_populism)              AS populism,
        any_value(speaker_anti_elitism)          AS anti_elitism,
        any_value(speaker_people_centrism)       AS people_centrism,
        any_value(speaker_anti_pluralism)        AS anti_pluralism,
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


PARTY_TARGETS_PARQUET = data.ANALYSIS_DIR / "target_party_resolution.parquet"


def load_party_panel(credit="bloc"):
    """PARTY-year panel: one row per (country, source_dataset, partyfacts_id, year).

    Built for the accusee-side hypotheses (H2b, H4a, H4c), which concern party
    attributes and are broken at the individual level: the government recode moved
    ~42k minister-targets out of the person-level DV, so governing-party MPs have
    artificially depleted individual received-counts.

        acc_received = acc_recv_person + acc_recv_party + acc_recv_bloc

    Three routes by which a party can be accused:
      acc_recv_person — a named MP of that party was the target
      acc_recv_party  — the party itself was named (resolve_party_targets.py)
      acc_recv_bloc   — "the government" for incumbents, "the opposition" for
                        the rest, DIVIDED by the number of parties on that side
                        so one accusation is not multiplied across coalition
                        partners

    The bloc term is what makes H4c testable: attacks on the executive are the
    single largest target category and are aimed at incumbents by definition, so
    excluding them would omit most of the evidence. It is symmetric because the
    opposition bloc is credited the same way.

    `credit`:
        "bloc"    — default, as above
        "none"    — named targets only (persons + parties); use to see how much
                    the bloc term is doing
        "pm_only" — government bloc to the PM's party alone, undivided
    """
    import numpy as np
    import pandas as pd

    df = load_panel()
    df = df.dropna(subset=["partyfacts_id"]).copy()
    df["partyfacts_id"] = df["partyfacts_id"].astype("int64")

    keys = ["country", "source_dataset", "partyfacts_id", "year"]
    pp = (df.groupby(keys, dropna=False)
            .agg(n_sentences=("n_sentences", "sum"),
                 n_mps=("speaker_id", "nunique"),
                 acc_made=("n_accusations_made", "sum"),
                 acc_recv_person=("n_accusations_received", "sum"),
                 party_name=("party_name", "first"),
                 populism=("populism", "mean"),
                 anti_elitism=("anti_elitism", "mean"),
                 people_centrism=("people_centrism", "mean"),
                 anti_pluralism=("anti_pluralism", "mean"),
                 left_right=("left_right", "mean"),
                 cultural_conservatism=("cultural_conservatism", "mean"),
                 in_cabinet=("in_cabinet", "max"),
                 is_pm_party=("is_pm_party", "max"),
                 vote_share_last=("vote_share_last", "mean"),
                 share_female=("female", lambda s: pd.to_numeric(s, errors="coerce").mean()),
                 mean_age=("age", "mean"),
                 mean_isced=("highest_isced", "mean"))
            .reset_index())

    # --- accusations aimed at this party by name ----------------------------
    if PARTY_TARGETS_PARQUET.exists():
        pt = pd.read_parquet(PARTY_TARGETS_PARQUET,
                             columns=["country", "source_dataset", "date",
                                      "target_partyfacts_id"])
        pt = pt.dropna(subset=["target_partyfacts_id"])
        pt["year"] = pd.to_numeric(pt["date"].astype(str).str[:4], errors="coerce")
        pt = pt.dropna(subset=["year"])
        pt["partyfacts_id"] = pt["target_partyfacts_id"].astype("int64")
        pt["year"] = pt["year"].astype("int64")
        pt = (pt.groupby(["country", "source_dataset", "partyfacts_id", "year"])
                .size().rename("acc_recv_party").reset_index())
        pp = pp.merge(pt, on=["country", "source_dataset", "partyfacts_id", "year"],
                      how="left")
        pp["acc_recv_party"] = pp["acc_recv_party"].fillna(0)
    else:
        print(f"note: {PARTY_TARGETS_PARQUET.name} not found -- run "
              f"party_linkage/resolve_party_targets.py; acc_recv_party = 0")
        pp["acc_recv_party"] = 0

    # --- bloc-directed accusations, by country-dataset-year -----------------
    con = data.duck()
    gov = con.execute("""
        SELECT country, source_dataset,
               CAST(substr(date, 1, 4) AS INT) AS year,
               COUNT(*) AS n_gov_targets
        FROM accusations
        WHERE target_type = 'government'
          AND date IS NOT NULL AND length(date) >= 4
        GROUP BY 1, 2, 3
    """).df()
    con.close()
    pp = pp.merge(gov, on=["country", "source_dataset", "year"], how="left")
    pp["n_gov_targets"] = pp["n_gov_targets"].fillna(0)

    # "the opposition" / "those opposite": the mirror of the government bloc,
    # picked up by resolve_party_targets.py when a party target had no name.
    if PARTY_TARGETS_PARQUET.exists():
        bl = pd.read_parquet(PARTY_TARGETS_PARQUET,
                             columns=["country", "source_dataset", "date",
                                      "target_bloc"])
        bl = bl[bl["target_bloc"].isin(["opposition", "government"])].copy()
        bl["year"] = pd.to_numeric(bl["date"].astype(str).str[:4], errors="coerce")
        bl = bl.dropna(subset=["year"])
        bl["year"] = bl["year"].astype("int64")
        wide = (bl.groupby(["country", "source_dataset", "year", "target_bloc"])
                  .size().unstack(fill_value=0).reset_index())
        wide = wide.rename(columns={"opposition": "n_opp_targets",
                                    "government": "n_gov_bloc_extra"})
        pp = pp.merge(wide, on=["country", "source_dataset", "year"], how="left")
    for col in ("n_opp_targets", "n_gov_bloc_extra"):
        if col not in pp.columns:
            pp[col] = 0
        pp[col] = pp[col].fillna(0)
    pp["n_gov_targets"] = pp["n_gov_targets"] + pp["n_gov_bloc_extra"]

    # --- bloc-directed accusations, split across the parties on each side ----
    # "the government" is an attack on the incumbents; "the opposition" on the
    # opposition. Each bloc count is DIVIDED by the number of parties on that
    # side, so a single accusation is not multiplied across coalition partners.
    grp = ["country", "source_dataset", "year"]
    n_gov_parties = (pp.assign(g=(pp["in_cabinet"] == 1).astype(int))
                       .groupby(grp)["g"].transform("sum"))
    n_opp_parties = (pp.assign(o=(pp["in_cabinet"] != 1).astype(int))
                       .groupby(grp)["o"].transform("sum"))

    in_gov = pp["in_cabinet"] == 1
    pp["acc_recv_bloc"] = np.where(
        in_gov,
        pp["n_gov_targets"] / n_gov_parties.replace(0, np.nan),
        pp["n_opp_targets"] / n_opp_parties.replace(0, np.nan),
    )
    pp["acc_recv_bloc"] = pp["acc_recv_bloc"].fillna(0)

    if credit == "bloc":                       # default: symmetric bloc credit
        bloc = pp["acc_recv_bloc"]
    elif credit == "pm_only":
        bloc = np.where(pp["is_pm_party"] == 1, pp["n_gov_targets"], 0)
    elif credit == "none":
        bloc = 0
    else:
        raise ValueError(f"unknown credit rule: {credit}")

    pp["acc_received"] = (pp["acc_recv_person"] + pp["acc_recv_party"] + bloc)
    pp["acc_received"] = pp["acc_received"].round().astype("int64")

    pp["log_exposure"] = np.log(pp["n_sentences"])
    pp["country_year"] = pp["country"] + "_" + pp["year"].astype(str)
    return pp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    build_panel(force=ap.parse_args().force)
