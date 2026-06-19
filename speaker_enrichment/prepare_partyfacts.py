"""
prepare_partyfacts.py
=====================
One-off prep for the party-annotation pipeline (Group B).

Reads the PartyFacts core-parties CSV, restricts it to our corpus countries,
drops parties that did not exist after 1990 and PartyFacts "technical" entries
(electoral alliances, "independents", "others"), and emits, per country, a
closed list of valid parties with their partyfacts_id. This list is injected
into the Group-B prompt so the model returns PartyFacts identifiers directly
instead of free-text names that would need fuzzy linking later.

Outputs (under PARTYFACTS_DIR):
  - partyfacts_filtered.csv : the filtered rows kept (traceability)
  - party_prompts.json      : {iso2: {"parties": [...], "prompt_block": "..."}}

A party is kept if it existed at any point after 1990:
    year_last >= 1990  OR  year_last is null (still active / unknown end)

Usage:
    python3 prepare_partyfacts.py
"""

import json
from pathlib import Path

import pandas as pd

PARTYFACTS_DIR = "/home/tom/data/partyfacts"
CORE_CSV       = f"{PARTYFACTS_DIR}/partyfacts-core-parties.csv"
FILTERED_CSV   = f"{PARTYFACTS_DIR}/partyfacts_filtered.csv"
PROMPTS_JSON   = f"{PARTYFACTS_DIR}/party_prompts.json"

CUTOFF_YEAR = 1990

# Our 34 corpus countries: ISO-2 (our data) -> ISO-3 (PartyFacts)
ISO2_TO_ISO3 = {
    "AT":"AUT","AU":"AUS","BA":"BIH","BE":"BEL","BG":"BGR","CA":"CAN","CZ":"CZE",
    "DE":"DEU","DK":"DNK","EE":"EST","ES":"ESP","FI":"FIN","FR":"FRA","GB":"GBR",
    "GR":"GRC","HR":"HRV","HU":"HUN","IS":"ISL","IT":"ITA","LT":"LTU","LV":"LVA",
    "NL":"NLD","NO":"NOR","NZ":"NZL","PL":"POL","PT":"PRT","RO":"ROU","RS":"SRB",
    "SE":"SWE","SI":"SVN","SK":"SVK","TR":"TUR","UA":"UKR","US":"USA",
}
ISO3_TO_ISO2 = {v: k for k, v in ISO2_TO_ISO3.items()}


def _clean(val) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s or None


def _years(year_first, year_last) -> str:
    a = "" if pd.isna(year_first) else str(int(year_first))
    b = "" if pd.isna(year_last) else str(int(year_last))
    if a and b:
        return f"{a}–{b}"
    if a:
        return f"{a}–"          # still active / unknown end
    if b:
        return f"–{b}"
    return "?"


def _display_name(row) -> str:
    """English name preferred; fall back to native name. Append native if both
    exist and differ, to help the model match CV mentions in any language."""
    eng    = _clean(row["name_english"])
    native = _clean(row["name"])
    if eng and native and eng != native:
        return f"{eng} ({native})"
    return eng or native or "(unknown)"


def main():
    df = pd.read_csv(CORE_CSV)
    iso3 = set(ISO2_TO_ISO3.values())

    ours = df[df["country"].isin(iso3)].copy()

    # post-1990: existed at any point after the cutoff
    post1990 = ours[(ours["year_last"] >= CUTOFF_YEAR) | (ours["year_last"].isna())]
    # drop PartyFacts "technical" non-party entries
    kept = post1990[post1990["technical"].fillna(False) == False].copy()

    kept.to_csv(FILTERED_CSV, index=False)
    print(f"Kept {len(kept):,} parties across {kept['country'].nunique()} countries")
    print(f"  filtered rows -> {FILTERED_CSV}")

    out: dict[str, dict] = {}
    for c3, grp in kept.groupby("country"):
        cc = ISO3_TO_ISO2[c3]
        # sort by abbrev then id for stable, readable prompts
        grp = grp.sort_values(["name_short", "partyfacts_id"], na_position="last")

        parties = []
        lines = []
        for _, r in grp.iterrows():
            pid    = int(r["partyfacts_id"])
            abbrev = _clean(r["name_short"])
            disp   = _display_name(r)
            years  = _years(r["year_first"], r["year_last"])
            parties.append({
                "partyfacts_id": pid,
                "abbrev":        abbrev,
                "name":          disp,
                "years":         years,
            })
            lines.append(f"{pid} | {abbrev or '-'} | {disp} | {years}")

        prompt_block = "\n".join(lines)
        out[cc] = {"parties": parties, "prompt_block": prompt_block}

    Path(PROMPTS_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print(f"  prompt blocks -> {PROMPTS_JSON}")

    # show a small + a mid-size country so the format is easy to eyeball
    for cc in ("US", "BE"):
        print(f"\n===== {cc} ({len(out[cc]['parties'])} parties) =====")
        print(out[cc]["prompt_block"])


if __name__ == "__main__":
    main()
