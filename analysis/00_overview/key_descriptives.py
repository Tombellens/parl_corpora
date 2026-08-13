"""
key_descriptives.py
===================
Headline descriptives for the paper / appendix, WITH the country exclusions
applied (i.e. the analysis corpus, not the raw pipeline output).

Run from the analysis/ folder:
    python3 00_overview/key_descriptives.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from lib import data
from lib.codebooks import EXCLUDED_COUNTRIES


def rule(t):
    print("\n" + "=" * 62 + f"\n{t}\n" + "=" * 62)


def main():
    con = data.duck()          # exclusions applied

    rule("CORPUS")
    corpus = con.execute("""
        SELECT COUNT(*)                          AS sentences,
               COUNT(DISTINCT source_speech_id)  AS speeches,
               COUNT(DISTINCT speaker)           AS speakers_in_corpus,
               COUNT(DISTINCT country)           AS countries,
               COUNT(DISTINCT source_dataset)    AS datasets,
               MIN(date) AS first_date, MAX(date) AS last_date
        FROM corpus
    """).df().T
    corpus.columns = [""]
    print(corpus.to_string())

    rule("ACCUSATIONS")
    acc = con.execute("""
        SELECT COUNT(*)                                                   AS accusations,
               COUNT(DISTINCT accuser_speaker_id)                         AS distinct_accusers,
               SUM(CASE WHEN is_interjection = 1 THEN 1 ELSE 0 END)       AS interjections,
               SUM(CASE WHEN accuser_match = 'resolved' THEN 1 ELSE 0 END) AS accuser_with_party
        FROM accusations
    """).df().T
    acc.columns = [""]
    print(acc.to_string())

    n_acc = con.execute("SELECT COUNT(*) FROM accusations").fetchone()[0]
    n_sent = con.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
    print(f"\naccusation rate: {n_acc / n_sent * 10_000:.1f} per 10,000 sentences")

    print("\ntarget type:")
    tt = con.execute("""
        SELECT target_type, COUNT(*) AS n
        FROM accusations GROUP BY 1 ORDER BY 2 DESC
    """).df()
    tt["%"] = (tt["n"] / tt["n"].sum() * 100).round(1)
    print(tt.to_string(index=False))

    rule("SPEAKERS AND BIOGRAPHICAL COVERAGE")
    # one row per distinct corpus speaker, using the accusation-side variables
    sp = con.execute("""
        SELECT DISTINCT speaker_speaker_id AS sid,
               speaker_gender        AS gender,
               speaker_birth_year    AS birth_year,
               speaker_highest_isced AS isced,
               speaker_career_sectors AS sectors,
               speaker_partyfacts_id AS pf
        FROM corpus
        WHERE speaker_speaker_id IS NOT NULL
    """).df()
    n_sp = sp["sid"].nunique()
    print(f"speakers linked to an enrichment record : {n_sp:,}")
    for label, col in [("gender", "gender"), ("birth year", "birth_year"),
                       ("education (ISCED)", "isced"),
                       ("career sectors", "sectors"),
                       ("party (PartyFacts id)", "pf")]:
        k = sp[col].notna().sum()
        print(f"  with {label:<24}: {k:>7,}  ({k / max(n_sp, 1) * 100:5.1f}%)")

    rule("PARTIES")
    parties = con.execute("""
        SELECT COUNT(DISTINCT speaker_partyfacts_id) AS parties_in_corpus,
               COUNT(DISTINCT CASE WHEN speaker_left_right IS NOT NULL
                                   THEN speaker_partyfacts_id END) AS with_vparty,
               COUNT(DISTINCT CASE WHEN speaker_vote_share_last IS NOT NULL
                                   THEN speaker_partyfacts_id END) AS with_parlgov
        FROM corpus
    """).df().T
    parties.columns = [""]
    print(parties.to_string())

    rule("BY COUNTRY")
    byc = con.execute("""
        WITH s AS (
            SELECT country, COUNT(*) AS sentences,
                   COUNT(DISTINCT source_speech_id) AS speeches,
                   COUNT(DISTINCT speaker) AS speakers
            FROM corpus GROUP BY 1
        ), a AS (
            SELECT country, COUNT(*) AS accusations FROM accusations GROUP BY 1
        )
        SELECT s.country, s.sentences, s.speeches, s.speakers,
               COALESCE(a.accusations, 0) AS accusations
        FROM s LEFT JOIN a USING (country) ORDER BY s.sentences DESC
    """).df()
    byc["per_10k"] = (byc["accusations"] / byc["sentences"] * 10_000).round(1)
    print(byc.to_string(index=False))
    print(f"\nexcluded from all of the above: {sorted(EXCLUDED_COUNTRIES)}")


if __name__ == "__main__":
    main()
