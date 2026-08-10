"""
feasibility_checks.py
=====================
Two blocking checks before any modelling:

  PART A — V-Party `left_right` (v2pariglef) POLARITY.
           H4a/H4b are pure sign predictions, so a reversed scale would reverse
           the headline finding. We do NOT assume a direction: we print the
           parties at each end of the scale and let the names settle it.
           Same treatment for `populism` as a sanity check on H2.

  PART B — H3d (retaliation) FEASIBILITY.
           Retaliation needs directed dyads: accuser AND target both resolved
           to individuals. This counts usable accusations, dyads, reciprocal
           dyads, and the timing of direction switches — i.e. whether H3d is a
           centerpiece or a footnote.

Run from the analysis/ folder:
    python3 00_overview/feasibility_checks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # analysis/ on path

import numpy as np
import pandas as pd

from lib import data

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

MIN_ACC_PER_PARTY = 200      # party must have this many accusations to be shown
RETAL_WINDOWS = [30, 90, 365]


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# PART A — scale polarity
# ---------------------------------------------------------------------------
def part_a():
    rule("PART A — V-Party scale polarity (left_right, populism)")

    cols = ["id", "country", "accuser_party_name", "accuser_left_right",
            "accuser_populism", "accuser_match"]
    acc = data.load_accusations(columns=cols)

    print(f"accusations loaded: {len(acc):,}")
    lr = acc["accuser_left_right"].dropna()
    pop = acc["accuser_populism"].dropna()
    print(f"\nleft_right : n={len(lr):,}  min={lr.min():.3f}  max={lr.max():.3f}  "
          f"mean={lr.mean():.3f}")
    print(f"             quantiles {np.round(lr.quantile([.05,.25,.5,.75,.95]).values, 3)}")
    print(f"populism   : n={len(pop):,}  min={pop.min():.3f}  max={pop.max():.3f}  "
          f"mean={pop.mean():.3f}")
    print(f"             quantiles {np.round(pop.quantile([.05,.25,.5,.75,.95]).values, 3)}")

    def party_table(var):
        t = (acc.dropna(subset=[var, "accuser_party_name"])
                .groupby(["country", "accuser_party_name"], as_index=False)
                .agg(value=(var, "mean"), n_accusations=("id", "size")))
        return t[t["n_accusations"] >= MIN_ACC_PER_PARTY].sort_values("value")

    t = party_table("accuser_left_right")
    print(f"\n--- left_right: 20 LOWEST-scoring parties "
          f"(>= {MIN_ACC_PER_PARTY} accusations) ---")
    print(t.head(20).to_string(index=False))
    print(f"\n--- left_right: 20 HIGHEST-scoring parties ---")
    print(t.tail(20).to_string(index=False))
    print("""
INTERPRET: if the HIGHEST block is dominated by radical-right parties (AfD, FPÖ,
Rassemblement National, Vlaams Belang, Fidesz, Lega ...) then HIGHER = RIGHT,
which is what H4a/H4b assume. If instead the highest block is socialist/green/
left parties, the scale is REVERSED and every left_right sign must be flipped
before interpretation.""")

    t = party_table("accuser_populism")
    print(f"\n--- populism: 20 HIGHEST-scoring parties ---")
    print(t.tail(20).to_string(index=False))
    print("""
INTERPRET: this block should be recognisably populist. If it is not, the
partyfacts -> V-Party join is suspect and H2 cannot be tested as specified.""")


# ---------------------------------------------------------------------------
# PART B — H3d feasibility
# ---------------------------------------------------------------------------
def part_b():
    rule("PART B — H3d retaliation feasibility")

    cols = ["id", "country", "source_dataset", "date", "target_type",
            "is_interjection", "accuser_speaker_id", "target_speaker_id"]
    acc = data.load_accusations(columns=cols)

    print(f"all accusations                     : {len(acc):,}")
    person = acc[acc["target_type"] == "person"]
    print(f"target_type == 'person'             : {len(person):,}")
    print(f"  accuser resolved                  : {person['accuser_speaker_id'].notna().sum():,}")
    print(f"  target  resolved                  : {person['target_speaker_id'].notna().sum():,}")

    d = person.dropna(subset=["accuser_speaker_id", "target_speaker_id"]).copy()
    d = d[d["accuser_speaker_id"] != d["target_speaker_id"]]        # drop self-accusations
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    print(f"  BOTH resolved, dated, non-self    : {len(d):,}   <-- usable events")

    if d.empty:
        print("\nNo usable directed events. H3d is not testable as specified.")
        return

    print(f"\ndistinct accusers                   : {d['accuser_speaker_id'].nunique():,}")
    print(f"distinct targets                    : {d['target_speaker_id'].nunique():,}")

    # ---- directed dyads -------------------------------------------------
    pairs = (d.groupby(["accuser_speaker_id", "target_speaker_id"])
               .size().rename("n").reset_index())
    pairset = set(zip(pairs["accuser_speaker_id"], pairs["target_speaker_id"]))
    print(f"\ndirected dyads (A->B)               : {len(pairs):,}")
    print(f"  events per directed dyad          : "
          f"mean {pairs['n'].mean():.2f}, median {pairs['n'].median():.0f}, "
          f"max {pairs['n'].max():,}")

    recip_directed = [(a, b) for (a, b) in pairset if (b, a) in pairset]
    n_unordered = len(recip_directed) // 2
    print(f"\nRECIPROCAL dyads (both directions)  : {n_unordered:,}  <-- H3d core sample")

    if n_unordered == 0:
        print("\nNo reciprocal dyads: within-dyad retaliation is not identifiable.")
        return

    cnt = {(a, b): n for a, b, n in
           zip(pairs["accuser_speaker_id"], pairs["target_speaker_id"], pairs["n"])}
    both2 = sum(1 for (a, b) in recip_directed
                if a < b and cnt.get((a, b), 0) >= 2 and cnt.get((b, a), 0) >= 2)
    print(f"  ... with >= 2 events EACH way     : {both2:,}  "
          f"(supports dyad fixed effects)")

    # ---- timing of direction switches -----------------------------------
    recip_set = set(recip_directed)
    dr = d[[(a, b) in recip_set for a, b in
            zip(d["accuser_speaker_id"], d["target_speaker_id"])]].copy()
    dr["dyad"] = [tuple(sorted((a, b))) for a, b in
                  zip(dr["accuser_speaker_id"], dr["target_speaker_id"])]
    dr = dr.sort_values(["dyad", "date"])

    gaps = []
    for _, g in dr.groupby("dyad", sort=False):
        acc_ids = g["accuser_speaker_id"].values
        dates = g["date"].values
        for i in range(len(g) - 1):
            # a direction switch = retaliation candidate
            if acc_ids[i] != acc_ids[i + 1]:
                gaps.append((dates[i + 1] - dates[i]) / np.timedelta64(1, "D"))
    gaps = np.array(gaps)

    print(f"\ndirection switches (A->B then B->A) : {len(gaps):,}")
    if len(gaps):
        print(f"  days between: median {np.median(gaps):.0f}, "
              f"p25 {np.percentile(gaps, 25):.0f}, p75 {np.percentile(gaps, 75):.0f}")
        for w in RETAL_WINDOWS:
            print(f"  within {w:>4} days: {(gaps <= w).sum():,} "
                  f"({(gaps <= w).mean()*100:.1f}%)")

    # ---- where the usable data lives ------------------------------------
    print("\nusable events by country (top 15):")
    print(d["country"].value_counts().head(15).to_string())
    print("\nusable events by source_dataset (top 15):")
    print(d["source_dataset"].value_counts().head(15).to_string())
    print(f"\ninterjections among usable events   : "
          f"{int(pd.to_numeric(d['is_interjection'], errors='coerce').fillna(0).sum()):,}")

    # ---- verdict --------------------------------------------------------
    rule("VERDICT")
    if n_unordered >= 2000 and both2 >= 500:
        print("STRONG — reciprocal dyads are plentiful. H3d can carry a within-dyad\n"
              "design with dyad fixed effects. Make it the centerpiece.")
    elif n_unordered >= 500:
        print("MODERATE — enough for a within-dyad design, but expect wide CIs.\n"
              "Report alongside the permutation baseline; treat as supporting evidence.")
    else:
        print("WEAK — too few reciprocal dyads for dyad fixed effects. Fall back to a\n"
              "country-level reciprocity test (is the accuser->target matrix more\n"
              "symmetric than chance?) and present H3d descriptively.")


if __name__ == "__main__":
    import argparse
    import contextlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "all"], default="all",
                    help="run only Part A (polarity) or Part B (H3d feasibility)")
    ap.add_argument("--out", default="feasibility_report.txt",
                    help="write the report here as well as to the terminal "
                         "(default: feasibility_report.txt)")
    args = ap.parse_args()

    def run():
        if args.part in ("a", "all"):
            part_a()
        if args.part in ("b", "all"):
            part_b()

    with open(args.out, "w") as fh:
        class _Tee:
            def write(self, s):
                sys.__stdout__.write(s)
                fh.write(s)
            def flush(self):
                sys.__stdout__.flush()
                fh.flush()
        with contextlib.redirect_stdout(_Tee()):
            run()

    print(f"\n[report written to {args.out}]", file=sys.__stderr__)
