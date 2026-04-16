"""
enrich_us_bioguide.py
=====================
Resolves US Congressional Record speaker strings to full names and metadata
using the Congress.gov Bioguide API.

Steps:
  1. Fetch all congress members from api.congress.gov (paginated)
  2. Build lookup: (LAST_NAME, state_abbrev) → list of member records with date ranges
  3. Parse each US speaker string → extract last name + state
  4. Match to lookup using speaker date range
  5. Update name_cleaned, and add bioguide_id, party, gender, birth_year columns

Usage:
    python3 enrich_us_bioguide.py
"""

import re
import time
import json
import os

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SPEAKER_NAMES_FILE = "/home/tom/data/speaker_names.csv"
BIOGUIDE_CACHE     = "/home/tom/data/bioguide_members.json"
API_KEY            = "3jZ9bJYaUr7Ur190xArVeqNMiEH0xWkIZMxdLw3i"
BASE_URL           = "https://api.congress.gov/v3"
PAGE_SIZE          = 250
RATE_LIMIT_DELAY   = 0.1   # seconds between requests

# State name → abbreviation
STATE_ABBREVS = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "puerto rico": "PR", "guam": "GU",
    "virgin islands": "VI", "american samoa": "AS",
    "northern mariana islands": "MP",
}


# ---------------------------------------------------------------------------
# 1. Fetch all members from Congress.gov API (with caching)
# ---------------------------------------------------------------------------

def fetch_all_members() -> list[dict]:
    """Download all congress members, using a local cache if available."""
    if os.path.exists(BIOGUIDE_CACHE):
        print(f"  Loading cached members from {BIOGUIDE_CACHE}...")
        with open(BIOGUIDE_CACHE) as f:
            members = json.load(f)
        print(f"  {len(members):,} members loaded from cache.")
        return members

    print("  Fetching all members from Congress.gov API...")
    members = []
    offset  = 0

    while True:
        for attempt in range(5):
            try:
                resp = requests.get(
                    f"{BASE_URL}/member",
                    params={"api_key": API_KEY, "limit": PAGE_SIZE, "offset": offset},
                    timeout=120,
                )
                resp.raise_for_status()
                break
            except Exception as e:
                print(f"\n  ⚠️  Attempt {attempt+1}/5 failed: {e}. Retrying...")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("Failed to fetch after 5 attempts.")
        data = resp.json()

        batch = data.get("members", [])
        if not batch:
            break
        members.extend(batch)

        total = data.get("pagination", {}).get("count", 0)
        print(f"  Fetched {len(members):,} / {total:,}...", end="\r")

        if len(members) >= total:
            break
        offset += PAGE_SIZE
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\n  Done. {len(members):,} members fetched.")

    # Cache for future runs
    with open(BIOGUIDE_CACHE, "w") as f:
        json.dump(members, f)
    print(f"  Cached to {BIOGUIDE_CACHE}")

    return members


# ---------------------------------------------------------------------------
# 2. Build lookup
# ---------------------------------------------------------------------------

def build_lookup(members: list[dict]) -> dict:
    """
    Returns {(LAST_NAME_UPPER, STATE_ABBREV): [member_dict, ...]}
    Each member_dict has: full_name, bioguide_id, party, state,
                          served_from (year), served_until (year)
    """
    lookup = {}

    for m in members:
        bioguide_id = m.get("bioguideId", "")
        name        = m.get("name", "")         # "Surname, Firstname"
        state       = m.get("state", "")
        party_name  = m.get("partyName", "")

        # Parse "Surname, Firstname Middle" → last/full
        if "," in name:
            parts     = name.split(",", 1)
            last      = parts[0].strip()
            first     = parts[1].strip()
            full_name = f"{first} {last}"
        else:
            last      = name.strip()
            full_name = name.strip()

        # Date range from terms
        terms = m.get("terms", {})
        if isinstance(terms, dict):
            term_list = terms.get("item", [])
        elif isinstance(terms, list):
            term_list = terms
        else:
            term_list = []

        years = []
        for t in term_list:
            start = t.get("startYear") or t.get("start", "")
            end   = t.get("endYear")   or t.get("end",   "")
            try:
                years.append(int(str(start)[:4]))
            except (ValueError, TypeError):
                pass
            try:
                years.append(int(str(end)[:4]))
            except (ValueError, TypeError):
                pass

        served_from  = min(years) if years else None
        served_until = max(years) if years else None

        record = {
            "full_name":    full_name,
            "bioguide_id":  bioguide_id,
            "party":        party_name,
            "state":        state,
            "served_from":  served_from,
            "served_until": served_until,
        }

        key = (last.upper(), state.upper())
        lookup.setdefault(key, []).append(record)

        # Also index without state for fallback
        key_no_state = (last.upper(), "")
        lookup.setdefault(key_no_state, []).append(record)

    return lookup


# ---------------------------------------------------------------------------
# 3. Parse US speaker strings
# ---------------------------------------------------------------------------

# Titles to strip
_TITLE_RE = re.compile(
    r"^(Mr\.|Ms\.|Mrs\.|Miss|Dr\.|Hon\.|the\s+)?",
    re.IGNORECASE,
)

# "of STATE" suffix
_OF_STATE_RE = re.compile(r"\s+of\s+(.+)$", re.IGNORECASE)

# Parenthetical (e.g. "The SPEAKER pro tempore (Mr. Hastert)")
_PAREN_RE = re.compile(r"\(([^)]+)\)")


def parse_speaker_string(raw: str):
    """
    Extract (last_name_upper, state_abbrev_or_empty) from a CR speaker string.
    Returns (None, None) if parsing fails.
    """
    s = raw.strip()

    # Handle "The X (Mr. NAME)" format — extract from parens
    paren = _PAREN_RE.search(s)
    if paren:
        s = paren.group(1).strip()

    # Strip title
    s = _TITLE_RE.sub("", s).strip()

    # Extract state
    state = ""
    m = _OF_STATE_RE.search(s)
    if m:
        state_str = m.group(1).strip().lower()
        state     = STATE_ABBREVS.get(state_str, state_str.upper()[:2])
        s         = s[: m.start()].strip()

    last = s.strip().upper()
    if not last or len(last) < 2:
        return None, None

    return last, state


def match_member(last: str, state: str, min_date: str, max_date: str,
                 lookup: dict) -> dict | None:
    """Find the best matching member record."""
    def year(date_str):
        try:
            return int(str(date_str)[:4])
        except (ValueError, TypeError):
            return None

    y_min = year(min_date)
    y_max = year(max_date)

    # Try with state first, then without
    for key in [(last, state.upper()), (last, "")]:
        candidates = lookup.get(key, [])
        if not candidates:
            continue

        # Filter by date overlap if we have years
        if y_min and y_max:
            date_filtered = [
                c for c in candidates
                if (c["served_from"] is None or c["served_from"] <= y_max) and
                   (c["served_until"] is None or c["served_until"] >= y_min)
            ]
            if date_filtered:
                candidates = date_filtered

        # If exactly one match, use it; if multiple, pick the longest-serving
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1 and state:
            # Prefer state match
            state_match = [c for c in candidates if c["state"].upper() == state.upper()]
            if state_match:
                return state_match[0]
            return candidates[0]

    return None


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {SPEAKER_NAMES_FILE}...")
    df = pd.read_csv(SPEAKER_NAMES_FILE, dtype=str)
    print(f"  {len(df):,} rows loaded.")

    # Add metadata columns if missing
    for col in ["bioguide_id", "us_party", "us_gender", "us_birth_year"]:
        if col not in df.columns:
            df[col] = None

    # Work only on US persons
    us_mask = (df["source_dataset_type"] == "congressional-record") & (df["is_person"] == "True")
    us_df   = df[us_mask]
    print(f"  {len(us_df):,} US person rows to enrich.")

    print("\nFetching Bioguide members...")
    members = fetch_all_members()

    print("\nBuilding name lookup...")
    lookup = build_lookup(members)
    print(f"  Lookup has {len(lookup):,} keys.")

    print("\nMatching speakers...")
    matched = 0
    ambiguous = 0
    unmatched = 0

    for idx, row in tqdm(us_df.iterrows(), total=len(us_df), desc="Matching"):
        speaker  = str(row["speaker"])
        last, state = parse_speaker_string(speaker)

        if not last:
            unmatched += 1
            continue

        member = match_member(last, state, row["min_date"], row["max_date"], lookup)

        if member:
            df.at[idx, "name_cleaned"]  = member["full_name"]
            df.at[idx, "bioguide_id"]   = member["bioguide_id"]
            df.at[idx, "us_party"]      = member["party"]
            matched += 1
        else:
            unmatched += 1

    print(f"\nMatching results:")
    print(f"  Matched:   {matched:,}")
    print(f"  Unmatched: {unmatched:,}")

    print(f"\nSaving to {SPEAKER_NAMES_FILE}...")
    df.to_csv(SPEAKER_NAMES_FILE, index=False)
    print("Done!")

    # Sample
    enriched = df[us_mask & df["bioguide_id"].notna()]
    print(f"\nSample enriched US speakers:")
    print(enriched[["speaker", "name_cleaned", "us_party", "bioguide_id"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
