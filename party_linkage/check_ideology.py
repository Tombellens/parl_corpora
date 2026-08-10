"""
check_ideology.py
=================
Pre-flight check on the V-Party measures BEFORE rebuilding the datasets.

Verifies that `cultural_conservatism` is oriented higher = more conservative
(a flipped sign would reverse H4a/H4b), and reports how separable the party
dimensions are. Uses the same loader as the builders, so there is one definition
of the index.

Usage:
    python3 check_ideology.py
"""

import pandas as pd

import config
from build_party_vars import load_vparty, VPARTY_MEASURES

pd.set_option("display.width", 170)


def main():
    print("loading V-Party ...")
    vp = load_vparty()

    # flatten {pfid: [(date, {measure: val})]} -> long frame
    rows = []
    for pfid, entries in vp.items():
        for d, rec in entries:
            rows.append({"partyfacts_id": pfid, "date": d, **rec})
    df = pd.DataFrame(rows)
    print(f"party-elections: {len(df):,}   parties: {df['partyfacts_id'].nunique():,}")

    # party names straight from the CSV
    raw = pd.read_csv(config.VPARTY_CSV, low_memory=False,
                      usecols=lambda c: c in {"pf_party_id", "v2paenname", "country_name"})
    raw = raw.dropna(subset=["pf_party_id"])
    raw["partyfacts_id"] = raw["pf_party_id"].astype(int)
    names = (raw.groupby("partyfacts_id")
                .agg(party=("v2paenname", "first"), country=("country_name", "first"))
                .reset_index())
    df = df.merge(names, on="partyfacts_id", how="left")

    print("\ncoverage per measure:")
    for m in VPARTY_MEASURES:
        print(f"  {m:<24} {df[m].notna().sum():>7,} "
              f"({df[m].notna().mean()*100:5.1f}%)")

    agg = (df.dropna(subset=["cultural_conservatism"])
             .groupby(["country", "party"], as_index=False)
             .agg(cultural=("cultural_conservatism", "mean"),
                  econ=("left_right", "mean"),
                  n=("party", "size")))
    agg = agg[agg["n"] >= 3].sort_values("cultural")

    print("\n--- MOST PROGRESSIVE (cultural_conservatism lowest) ---")
    print(agg.head(15).to_string(index=False))
    print("\n--- MOST CONSERVATIVE (cultural_conservatism highest) ---")
    print(agg.tail(15).to_string(index=False))
    print("\nEXPECT the conservative end to be radical-right / religious parties.\n"
          "If greens and social democrats are there instead, the sign is flipped.")

    print("\n--- correlations between party dimensions ---")
    print(df[VPARTY_MEASURES].corr().round(3).to_string())
    print("\nA low econ x cultural correlation means the two ideology dimensions are\n"
          "separable and H4a/H4b can be tested on each.")


if __name__ == "__main__":
    main()
