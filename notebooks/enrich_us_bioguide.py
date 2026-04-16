"""
enrich_us_bioguide.py
=====================
Resolves US Congressional Record speaker strings to full names and metadata
using the Congress.gov Bioguide API, with LLM disambiguation for uncertain cases.

Steps:
  1. Fetch all congress members per-congress from api.congress.gov (cached)
  2. Build lookup: (LAST_NAME, STATE) → list of member records with date ranges
  3. Parse each US speaker string → extract last name + state
  4. Pass 1: confident single-match cases resolved directly
  5. Pass 2: ambiguous/unmatched cases sent to LM Studio with top-3 candidates
  6. Update name_cleaned, bioguide_id, party columns

Usage:
    python3 enrich_us_bioguide.py
"""

import re
import time
import json
import os

import pandas as pd
import requests
from openai import OpenAI
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

# CR data covers congresses 103 (1993) through 118 (2023)
CR_CONGRESSES      = list(range(103, 119))

# LM Studio
LM_BASE_URL        = "http://localhost:1234/v1"
LM_MODEL           = "openai/gpt-oss-20b"
LM_BATCH_SIZE      = 10   # speakers per LLM call
LM_MAX_TOKENS      = 8000

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

def _api_get(url: str, params: dict) -> dict:
    """GET with retry logic."""
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"\n  ⚠️  Attempt {attempt+1}/5 failed: {e}. Retrying...")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after 5 attempts.")


def fetch_congress_members(congress: int) -> list[dict]:
    """Fetch all members for a specific congress number."""
    members = []
    offset  = 0
    url     = f"{BASE_URL}/member/congress/{congress}"
    while True:
        data  = _api_get(url, {"api_key": API_KEY, "limit": PAGE_SIZE, "offset": offset})
        batch = data.get("members", [])
        if not batch:
            break
        members.extend(batch)
        total = data.get("pagination", {}).get("count", 0)
        if len(members) >= total:
            break
        offset += PAGE_SIZE
        time.sleep(RATE_LIMIT_DELAY)
    return members


def fetch_all_members() -> list[dict]:
    """Download all congress members for CR congresses, using a local cache if available."""
    if os.path.exists(BIOGUIDE_CACHE):
        print(f"  Loading cached members from {BIOGUIDE_CACHE}...")
        with open(BIOGUIDE_CACHE) as f:
            members = json.load(f)
        print(f"  {len(members):,} members loaded from cache.")
        return members

    print(f"  Fetching members for congresses {CR_CONGRESSES[0]}–{CR_CONGRESSES[-1]}...")
    seen_ids = set()
    members  = []

    for congress in CR_CONGRESSES:
        batch = fetch_congress_members(congress)
        new   = [m for m in batch if m.get("bioguideId") not in seen_ids]
        for m in new:
            seen_ids.add(m.get("bioguideId"))
        members.extend(new)
        print(f"  Congress {congress}: {len(batch):,} members "
              f"({len(new):,} new) → total {len(members):,}")
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\n  Done. {len(members):,} unique members fetched.")

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


def get_candidates(last: str, state: str, min_date: str, max_date: str,
                   lookup: dict, top_n: int = 3) -> tuple[dict | None, list[dict]]:
    """
    Returns (confident_match_or_None, top_candidates).

    confident_match: set when there is exactly one date-filtered, state-confirmed
                     candidate — safe to use without LLM.
    top_candidates:  up to top_n candidates for the LLM to choose from (may be
                     empty, meaning no name match at all in the lookup).
    """
    def year(date_str):
        try:
            return int(str(date_str)[:4])
        except (ValueError, TypeError):
            return None

    y_min = year(min_date)
    y_max = year(max_date)

    # Collect all candidates by last name (state-filtered first, then all)
    state_key    = (last, state.upper())
    no_state_key = (last, "")

    state_cands    = lookup.get(state_key, [])
    all_name_cands = lookup.get(no_state_key, [])  # superset (includes state matches)

    def date_filter(cands):
        if not (y_min and y_max):
            return cands
        return [
            c for c in cands
            if (c["served_from"]  is None or c["served_from"]  <= y_max) and
               (c["served_until"] is None or c["served_until"] >= y_min)
        ]

    state_date = date_filter(state_cands)
    all_date   = date_filter(all_name_cands)

    # --- Confident match: exactly one date-filtered state match ---
    if len(state_date) == 1:
        return state_date[0], state_date

    # Rank candidates: state+date > state > date > all
    seen = set()
    ranked = []
    for c in (state_date + state_cands + all_date + all_name_cands):
        bid = c["bioguide_id"]
        if bid not in seen:
            seen.add(bid)
            ranked.append(c)

    top = ranked[:top_n]
    return None, top


# ---------------------------------------------------------------------------
# 3b. LLM disambiguation for uncertain cases
# ---------------------------------------------------------------------------

_LLM_SYSTEM = """You are resolving speaker strings from the US Congressional Record to Bioguide member records.

For each speaker, you are given:
- The raw speaker string and the year range it appears
- Up to 3 Bioguide candidates (letter a/b/c), each with full name, party, state, and years served

Your task: pick the best matching candidate, or reply "none" if no candidate fits.
Also decide if the speaker string is a real person at all (procedural roles like "The SPEAKER", "The CLERK", "A SENATOR" are NOT persons).

Respond ONLY with a JSON array, one object per input, in the same order:
{"is_person": true/false, "selected": "a"/"b"/"c"/"none", "confidence": "high"/"low"}

Rules:
- If is_person is false, set selected to "none"
- Prefer candidates whose served years overlap the appearance years
- State match is strong evidence; last name + state + year overlap = very confident
- If no candidates are provided and the string looks like a real name, is_person=true, selected="none"
- Do not invent bioguide IDs or names"""


def _fmt_candidate(letter: str, c: dict) -> str:
    served = f"{c['served_from'] or '?'}–{c['served_until'] or '?'}"
    return (f"  {letter}) {c['full_name']} | {c['party']} | {c['state']} "
            f"| served {served} | [{c['bioguide_id']}]")


def resolve_with_llm(client: OpenAI,
                     items: list[dict]) -> list[dict]:
    """
    items: list of {"speaker": str, "min_date": str, "max_date": str,
                     "candidates": [member_dict, ...]}
    Returns list of {"is_person": bool, "selected": member_dict | None}
    """
    letters = ["a", "b", "c"]

    lines = []
    for i, item in enumerate(items, 1):
        y0 = str(item["min_date"])[:4]
        y1 = str(item["max_date"])[:4]
        lines.append(f'{i}. "{item["speaker"]}" ({y0}–{y1})')
        if item["candidates"]:
            for j, c in enumerate(item["candidates"]):
                lines.append(_fmt_candidate(letters[j], c))
        else:
            lines.append("  (no candidates found in Bioguide)")

    prompt = "Resolve these Congressional Record speakers:\n\n" + "\n".join(lines)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=LM_MODEL,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=LM_MAX_TOKENS,
                temperature=0,
            )
            content = resp.choices[0].message.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            results = json.loads(content)
            if len(results) == len(items):
                # Map letter → candidate dict
                out = []
                for res, item in zip(results, items):
                    sel_letter = res.get("selected", "none")
                    cands = item["candidates"]
                    if sel_letter in letters and letters.index(sel_letter) < len(cands):
                        sel = cands[letters.index(sel_letter)]
                    else:
                        sel = None
                    out.append({
                        "is_person": res.get("is_person", True),
                        "selected":  sel,
                    })
                return out
            print(f"  ⚠️  Got {len(results)} results for {len(items)} items, retrying...")
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"  ⚠️  LLM error (attempt {attempt+1}): {e}")
            time.sleep(2)

    # Fallback: return no match for all
    return [{"is_person": True, "selected": None} for _ in items]


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {SPEAKER_NAMES_FILE}...")
    df = pd.read_csv(SPEAKER_NAMES_FILE, dtype=str)
    print(f"  {len(df):,} rows loaded.")

    # Add metadata columns if missing
    for col in ["bioguide_id", "us_party"]:
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

    # -----------------------------------------------------------------------
    # Build items: every US person speaker gets top-3 Bioguide candidates
    # -----------------------------------------------------------------------
    print("\nBuilding candidate lists...")
    items = []
    idxs  = []
    for idx, row in tqdm(us_df.iterrows(), total=len(us_df), desc="Candidates"):
        speaker = str(row["speaker"])
        last, state = parse_speaker_string(speaker)
        _, candidates = get_candidates(last, state, row["min_date"], row["max_date"], lookup) if last else (None, [])
        items.append({
            "speaker":    speaker,
            "min_date":   row["min_date"],
            "max_date":   row["max_date"],
            "candidates": candidates,
        })
        idxs.append(idx)

    # -----------------------------------------------------------------------
    # LLM resolution — all speakers, batched
    # -----------------------------------------------------------------------
    print("\nLLM resolution...")
    client = OpenAI(base_url=LM_BASE_URL, api_key="lm-studio")
    llm_matched    = 0
    llm_no_match   = 0
    llm_not_person = 0

    batches     = [items[i : i + LM_BATCH_SIZE] for i in range(0, len(items), LM_BATCH_SIZE)]
    idx_batches = [idxs[i  : i + LM_BATCH_SIZE] for i in range(0, len(idxs),  LM_BATCH_SIZE)]

    for batch_items, batch_idxs in tqdm(zip(batches, idx_batches),
                                         total=len(batches), desc="LLM batches"):
        results = resolve_with_llm(client, batch_items)
        for idx, res in zip(batch_idxs, results):
            if not res["is_person"]:
                df.at[idx, "is_person"] = "False"
                llm_not_person += 1
            elif res["selected"]:
                m = res["selected"]
                df.at[idx, "name_cleaned"] = m["full_name"]
                df.at[idx, "bioguide_id"]  = m["bioguide_id"]
                df.at[idx, "us_party"]     = m["party"]
                llm_matched += 1
            else:
                llm_no_match += 1

    print(f"\nMatching results:")
    print(f"  LLM matched:        {llm_matched:,}")
    print(f"  LLM → not a person: {llm_not_person:,}")
    print(f"  Still unmatched:    {llm_no_match:,}")
    print(f"  Total matched:      {llm_matched:,} / {len(us_df):,} "
          f"({llm_matched / len(us_df) * 100:.1f}%)")

    # Refresh mask after LLM may have flipped some is_person
    us_mask_final = (df["source_dataset_type"] == "congressional-record") & (df["is_person"] == "True")

    print(f"\nSaving to {SPEAKER_NAMES_FILE}...")
    df.to_csv(SPEAKER_NAMES_FILE, index=False)
    print("Done!")

    # Sample + coverage summary
    enriched = df[us_mask_final & df["bioguide_id"].notna()]
    print(f"\nSample enriched US speakers:")
    print(enriched[["speaker", "name_cleaned", "us_party", "bioguide_id"]].head(10).to_string(index=False))

    print(f"\nCoverage summary (US persons):")
    print(f"  Total US persons:      {us_mask_final.sum():,}")
    print(f"  Matched (bioguide_id): {df[us_mask_final]['bioguide_id'].notna().sum():,}")


if __name__ == "__main__":
    main()
