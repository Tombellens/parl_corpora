"""
resolve_party_targets.py
========================
Resolve accusations whose TARGET is a political party to a PartyFacts id.

Why: the accusee-side hypotheses (H2b, H4a, H4c) were only testable through
accusations aimed at named *individuals*, which is asymmetric — "the government"
can be attacked as a bloc but an opposition party could only be reached through
its MPs. Party-directed accusations are symmetric: governing and opposition
parties are equally nameable.

Matching is string-based, no model: the corpus is English-translated and
PartyFacts carries English names plus abbreviations, so exact and near-exact
matching gets most of the way.

  1. exact    — normalised target text equals a party name/abbreviation
  2. contains — a party name appears as a whole phrase inside the target text
  3. fuzzy    — best difflib ratio >= FUZZY_MIN and clear of the runner-up

Parties are restricted to the accusation's country and to parties active in the
accusation year.

Output: parquet with one row per resolved accusation.

Usage:
    python3 resolve_party_targets.py [--limit N]
"""

import argparse
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

import config

FILTERED_CSV = f"{config.DATA_DIR}/partyfacts/partyfacts_filtered.csv"
OUT_PARQUET = f"{config.ANALYSIS_DIR}/target_party_resolution.parquet"

FUZZY_MIN = 0.88
FUZZY_MARGIN = 0.04
MIN_KEY_LEN = 4          # shorter keys are too collision-prone for `contains`

ISO2_TO_ISO3 = {
    "AT": "AUT", "AU": "AUS", "BA": "BIH", "BE": "BEL", "BG": "BGR", "CA": "CAN",
    "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST", "ES": "ESP", "FI": "FIN",
    "FR": "FRA", "GB": "GBR", "GR": "GRC", "HR": "HRV", "HU": "HUN", "IS": "ISL",
    "IT": "ITA", "LT": "LTU", "LV": "LVA", "NL": "NLD", "NO": "NOR", "NZ": "NZL",
    "PL": "POL", "PT": "PRT", "RO": "ROU", "RS": "SRB", "SE": "SWE", "SI": "SVN",
    "SK": "SVK", "TR": "TUR", "UA": "UKR", "US": "USA",
}

# Words that carry no discriminating information in a party name.
STOP = {"the", "party", "parties", "of", "and", "for", "a", "an"}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def norm(s):
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _PUNCT.sub(" ", s.lower())
    return _WS.sub(" ", s).strip()


def content_key(s):
    """Normalised form with uninformative words removed."""
    return " ".join(w for w in norm(s).split() if w not in STOP)


def build_party_keys():
    """iso2 -> list of (partyfacts_id, key, display, year_first, year_last)."""
    pf = pd.read_csv(FILTERED_CSV)
    iso3_to_iso2 = {v: k for k, v in ISO2_TO_ISO3.items()}
    by_country = defaultdict(list)

    for r in pf.itertuples(index=False):
        cc = iso3_to_iso2.get(r.country)
        if cc is None:
            continue
        pid = int(r.partyfacts_id)
        display = r.name_english if isinstance(r.name_english, str) else r.name
        keys = set()
        for raw in (r.name_english, r.name, r.name_short):
            k = content_key(raw)
            if len(k) >= 3:
                keys.add(k)
            k2 = norm(raw)
            if len(k2) >= 3:
                keys.add(k2)
        for k in keys:
            by_country[cc].append((pid, k, display, r.year_first, r.year_last))
    return by_country


def _active(year, y0, y1):
    if year is None:
        return True
    if pd.notna(y0) and year < y0 - 1:
        return False
    if pd.notna(y1) and year > y1 + 2:      # small slack for lagged mentions
        return False
    return True


def resolve_one(text, country, year, by_country):
    cands = by_country.get(country, [])
    if not text or not cands:
        return None, None, "no_candidates"

    t_full, t_key = norm(text), content_key(text)
    live = [(pid, k, disp) for pid, k, disp, y0, y1 in cands if _active(year, y0, y1)]
    if not live:
        return None, None, "no_active_party"

    # 1. exact
    hits = {pid for pid, k, _ in live if k and (k == t_key or k == t_full)}
    if len(hits) == 1:
        pid = hits.pop()
        disp = next(d for p, _, d in live if p == pid)
        return pid, disp, "exact"
    if len(hits) > 1:
        return None, None, "ambiguous_exact"

    # 2. containment — party name appears as a whole phrase in the target
    hits = {pid for pid, k, _ in live
            if len(k) >= MIN_KEY_LEN and re.search(rf"\b{re.escape(k)}\b", t_key)}
    if len(hits) == 1:
        pid = hits.pop()
        disp = next(d for p, _, d in live if p == pid)
        return pid, disp, "contains"
    if len(hits) > 1:
        return None, None, "ambiguous_contains"

    # 3. fuzzy on the content key
    if not t_key:
        return None, None, "unresolved"
    scored = []
    for pid, k, disp in live:
        if len(k) < MIN_KEY_LEN:
            continue
        scored.append((SequenceMatcher(None, t_key, k).ratio(), pid, disp))
    if not scored:
        return None, None, "unresolved"
    scored.sort(reverse=True)
    best = scored[0]
    if best[0] >= FUZZY_MIN:
        rival = next((s for s in scored if s[1] != best[1]), None)
        if rival is None or (best[0] - rival[0]) >= FUZZY_MARGIN:
            return best[1], best[2], "fuzzy"
        return None, None, "ambiguous_fuzzy"
    return None, None, "unresolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import duckdb
    con = duckdb.connect()
    q = f"""
        SELECT id, country, source_dataset, date, target_text
        FROM read_parquet('{config.ACCUSATION_PARQUET}')
        WHERE target_type = 'political_party'
          AND target_text IS NOT NULL AND target_text <> ''
    """ + (f" LIMIT {args.limit}" if args.limit else "")
    df = con.execute(q).df()
    con.close()
    print(f"party-target accusations: {len(df):,}")

    by_country = build_party_keys()
    print(f"party lists loaded for {len(by_country)} countries "
          f"({sum(len(v) for v in by_country.values()):,} name keys)")

    df["year"] = pd.to_numeric(df["date"].astype(str).str[:4], errors="coerce")

    # cache: the same text repeats constantly within a country
    cache, out = {}, []
    for r in df.itertuples(index=False):
        key = (r.country, r.target_text, r.year)
        if key not in cache:
            cache[key] = resolve_one(r.target_text, r.country, r.year, by_country)
        pid, disp, how = cache[key]
        out.append((r.id, r.country, r.source_dataset, r.date, r.target_text,
                    pid, disp, how))

    res = pd.DataFrame(out, columns=["id", "country", "source_dataset", "date",
                                     "target_text", "target_partyfacts_id",
                                     "matched_name", "match_method"])

    Path(config.ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_PARQUET, index=False)

    n_ok = int(res["target_partyfacts_id"].notna().sum())
    print(f"\nresolved: {n_ok:,} of {len(res):,} ({n_ok/max(len(res),1)*100:.1f}%)")
    print("\nby method:")
    print(res["match_method"].value_counts().to_string())
    print("\nresolution rate by country:")
    by_c = (res.assign(ok=res["target_partyfacts_id"].notna())
               .groupby("country")
               .agg(n=("id", "size"), resolved=("ok", "sum")))
    by_c["rate"] = (by_c["resolved"] / by_c["n"] * 100).round(1)
    print(by_c.sort_values("n", ascending=False).head(20).to_string())
    print(f"\nwrote {OUT_PARQUET}")

    print("\nmost common unresolved target texts:")
    unres = res[res["target_partyfacts_id"].isna()]
    print(unres["target_text"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
