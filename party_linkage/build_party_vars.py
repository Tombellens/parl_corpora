"""
build_party_vars.py
===================
Cluster 1-3 party variables per accusation, keyed to the ACCUSER's party at the
accusation date.

Linkage chain:
  accuser (corpus speaker; for interjections the resolved interjector)
    -> speaker_names.csv -> name_cleaned
    -> speaker_enrichment.speakers -> speaker_id
    -> Group B annotation -> party (partyfacts_id) active in the accusation year
    -> ParlGov (via crosswalk dataset_party_id) : vote_share_last/avg/avg_last3/
       delta, in_cabinet, is_pm_party, years_since_government
    -> V-Party (via pf_party_id) : left_right (v2pariglef), populism (v2xpa_popul),
       anti_elitism (v2paanteli), people_centrism (v2papeople)

Writes a party_vars table (one row per accusation) into the accusations DB.

Usage:
    python3 build_party_vars.py [--limit N]
"""

import argparse
import bisect
import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# V-Party measures carried into the datasets, in output order.
# ---------------------------------------------------------------------------
VPARTY_MEASURES = [
    "left_right",             # v2pariglef — ECONOMIC left-right (higher = right)
    "populism",               # v2xpa_popul
    "anti_elitism",           # v2paanteli
    "people_centrism",        # v2papeople
    "cultural_conservatism",  # index of the 5 items below (higher = conservative)
    "anti_pluralism",         # v2xpa_antiplural (called v2xpa_illiberal in v1)
    "demonize_opponents",     # v2paopresp
]

# Cultural dimension item set follows Medzihorsky & Lindberg (2024, Party
# Politics). Every item is coded LOW = conservative/opposed, HIGH = progressive/
# supportive, so the index is sign-flipped to read higher = more conservative.
CULTURAL_ITEMS = ["v2paimmig", "v2palgbt", "v2paculsup", "v2parelig", "v2pawomlab"]

_DIRECT = {
    "left_right":         "v2pariglef",
    "populism":           "v2xpa_popul",
    "anti_elitism":       "v2paanteli",
    "people_centrism":    "v2papeople",
    "demonize_opponents": "v2paopresp",
}


# ---------------------------------------------------------------------------
# Loading the reference data
# ---------------------------------------------------------------------------
def _d(s):
    """ISO date string (first 10 chars) or None."""
    if not s or (isinstance(s, float) and pd.isna(s)):
        return None
    return str(s)[:10]


def load_crosswalk_parlgov():
    """partyfacts_id -> ParlGov party_id (int)."""
    x = pd.read_csv(config.CROSSWALK_CSV, low_memory=False)
    pg = x[x["dataset_key"] == "parlgov"].dropna(subset=["partyfacts_id"])
    out = {}
    for r in pg.itertuples(index=False):
        try:
            out[int(r.partyfacts_id)] = int(r.dataset_party_id)
        except (ValueError, TypeError):
            continue
    return out


def load_elections():
    """ParlGov party_id -> sorted [(date, vote_share)] for national elections."""
    e = pd.read_csv(config.ELECTION_CSV, low_memory=False)
    e = e[e["election_type"] == config.NATIONAL_ELECTION]
    by = {}
    for r in e.itertuples(index=False):
        d = _d(r.election_date)
        if d is None:
            continue
        by.setdefault(int(r.party_id), []).append((d, r.vote_share))
    for pid in by:
        by[pid].sort(key=lambda t: t[0])
    return by


def load_cabinets():
    """Per-country cabinet timelines. Returns:
      by_country : {country_id: (starts, gov_sets, pm)} each sorted by start_date
      party2country : {parlgov party_id: country_id}
    The lookup MUST be country-scoped — a global timeline would pick another
    country's cabinet as 'active' and wrongly report the party out of government.
    """
    c = pd.read_csv(config.CABINET_CSV, low_memory=False)
    cabs = {}                     # (country_id, cabinet_id) -> {start, gov, pm}
    party2country = {}
    for r in c.itertuples(index=False):
        d = _d(r.start_date)
        if d is None:
            continue
        cid, cabid, pid = int(r.country_id), int(r.cabinet_id), int(r.party_id)
        party2country[pid] = cid
        rec = cabs.setdefault((cid, cabid), {"start": d, "gov": set(), "pm": None})
        if int(r.cabinet_party or 0) == 1:
            rec["gov"].add(pid)
        if int(r.prime_minister or 0) == 1:
            rec["pm"] = pid
    grouped = {}
    for (cid, _cabid), rec in cabs.items():
        grouped.setdefault(cid, []).append(rec)
    by_country = {}
    for cid, lst in grouped.items():
        lst.sort(key=lambda v: v["start"])
        by_country[cid] = ([v["start"] for v in lst],
                           [v["gov"] for v in lst],
                           [v["pm"] for v in lst])
    return by_country, party2country


def load_vparty():
    """partyfacts_id -> sorted [(date, {measure: value})] for VPARTY_MEASURES."""
    v = pd.read_csv(config.VPARTY_CSV, low_memory=False)
    have = set(v.columns)

    def num(col):
        return (pd.to_numeric(v[col], errors="coerce") if col in have
                else pd.Series(np.nan, index=v.index))

    vals = {m: num(src) for m, src in _DIRECT.items()}
    # anti-pluralism was renamed between V-Party v1 and v2
    vals["anti_pluralism"] = num("v2xpa_antiplural" if "v2xpa_antiplural" in have
                                 else "v2xpa_illiberal")

    # cultural index: mean of z-scored items, flipped so higher = conservative
    items = [c for c in CULTURAL_ITEMS if c in have]
    if items:
        z = pd.DataFrame({c: num(c) for c in items})
        z = (z - z.mean()) / z.std()
        cult = -z.mean(axis=1)
        cult[z.notna().sum(axis=1) < 3] = np.nan      # need >= 3 of the 5 items
        vals["cultural_conservatism"] = cult
        print(f"  cultural index from {len(items)}/5 items, "
              f"{int(cult.notna().sum()):,} party-elections covered")
    else:
        vals["cultural_conservatism"] = pd.Series(np.nan, index=v.index)
        print("  WARNING: no cultural items present — cultural_conservatism is null")

    missing = [m for m in VPARTY_MEASURES if vals[m].notna().sum() == 0]
    if missing:
        print(f"  WARNING: no data for {missing}")

    pf = pd.to_numeric(v["pf_party_id"], errors="coerce")
    dates = v["historical_date"].astype(str).str[:10]
    by = {}
    for i in range(len(v)):
        if pd.isna(pf.iat[i]):
            continue
        rec = {m: (None if pd.isna(vals[m].iat[i]) else float(vals[m].iat[i]))
               for m in VPARTY_MEASURES}
        by.setdefault(int(pf.iat[i]), []).append((dates.iat[i], rec))
    for k in by:
        by[k].sort(key=lambda t: (t[0] or ""))
    return by


def load_bridge():
    """(speaker, country, source_dataset) -> [party dict...] via name_cleaned+enrichment."""
    sn = pd.read_csv(config.SPEAKER_NAMES, dtype=str).fillna("")
    # (name_cleaned, country, source_dataset) -> speaker_id
    con = sqlite3.connect(config.ENRICH_DB)
    sp = pd.read_sql_query(
        "SELECT speaker_id, name_cleaned, country, source_dataset FROM speakers", con)
    name2id = {(r.name_cleaned, r.country, r.source_dataset): r.speaker_id
               for r in sp.itertuples(index=False)}
    # speaker_id -> parties list (Group B)
    gb = pd.read_sql_query(
        "SELECT speaker_id, annotation_json FROM speaker_annotations "
        "WHERE group_name='B' AND status='success'", con)
    con.close()
    id2parties = {}
    for r in gb.itertuples(index=False):
        try:
            id2parties[r.speaker_id] = (json.loads(r.annotation_json) or {}).get("parties", [])
        except Exception:
            id2parties[r.speaker_id] = []

    bridge = {}
    for r in sn.itertuples(index=False):
        sid = name2id.get((r.name_cleaned, r.country, r.source_dataset))
        if sid is None:
            continue
        parties = id2parties.get(sid)
        if parties:
            bridge[(r.speaker, r.country, r.source_dataset)] = parties
    return bridge


# ---------------------------------------------------------------------------
# Per-accusation computation
# ---------------------------------------------------------------------------
def party_at_year(parties, year):
    cands = []
    for p in parties:
        s, e = p.get("start_year"), p.get("end_year")
        if (s is None or s <= year) and (e is None or e >= year):
            cands.append(p)
    if cands:
        return max(cands, key=lambda p: (p.get("start_year") or -9999))
    return parties[0] if parties else None


def election_vars(pid, D, elec):
    prior = [t for t in elec.get(pid, []) if t[0] <= D]
    if not prior:
        return (None, None, None, None)
    shares = [s for _, s in prior if s is not None]
    last = prior[-1][1]
    prev = prior[-2][1] if len(prior) >= 2 else None
    delta = (last - prev) if (last is not None and prev is not None) else None
    avg = sum(shares) / len(shares) if shares else None
    last3 = [s for _, s in prior[-3:] if s is not None]
    avg3 = sum(last3) / len(last3) if last3 else None
    return (last, avg, avg3, delta)


def cabinet_vars(pid, D, cab_by_country, party2country):
    cid = party2country.get(pid)
    if cid is None or cid not in cab_by_country:
        return (None, None, None)
    starts, gov_sets, pm = cab_by_country[cid]
    i = bisect.bisect_right(starts, D) - 1
    if i < 0:
        return (None, None, None)
    in_cab = 1 if pid in gov_sets[i] else 0
    is_pm = 1 if pm[i] == pid else 0
    if in_cab:
        return (1, is_pm, 0.0)
    j = i
    while j >= 0 and pid not in gov_sets[j]:
        j -= 1
    if j < 0:
        return (0, is_pm, None)          # never in government before D (this country)
    exit_date = starts[j + 1]            # first cabinet after the last gov spell
    yrs = (date.fromisoformat(D) - date.fromisoformat(exit_date)).days / 365.25
    return (0, is_pm, round(yrs, 2))


def vparty_vars(pfid, D, vp):
    """{measure: value} from the party's most recent election on/before D."""
    rows = vp.get(pfid, [])
    if not rows:
        return {m: None for m in VPARTY_MEASURES}
    prior = [r for r in rows if r[0] and r[0] <= D]
    return (prior[-1] if prior else rows[0])[1]


SCHEMA = """
CREATE TABLE IF NOT EXISTS party_vars (
    accusation_id           INTEGER PRIMARY KEY,
    partyfacts_id           INTEGER,
    party_match             TEXT,       -- resolved | no_speaker | no_party
    has_parlgov             INTEGER,
    has_vparty              INTEGER,
    vote_share_last         REAL,
    vote_share_avg          REAL,
    vote_share_avg_last3    REAL,
    vote_share_delta        REAL,
    in_cabinet              INTEGER,
    is_pm_party             INTEGER,
    years_since_government  REAL,
    left_right              REAL,
    populism                REAL,
    anti_elitism            REAL,
    people_centrism         REAL,
    cultural_conservatism   REAL,
    anti_pluralism          REAL,
    demonize_opponents      REAL
);
"""

N_NULL_TAIL = 7 + len(VPARTY_MEASURES)   # election(4) + cabinet(3) + V-Party


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("Loading reference data ...")
    xwalk = load_crosswalk_parlgov()
    elec  = load_elections()
    cab_by_country, party2country = load_cabinets()
    vp    = load_vparty()
    bridge = load_bridge()
    print(f"  crosswalk parlgov: {len(xwalk):,} | election parties: {len(elec):,} | "
          f"cabinet countries: {len(cab_by_country):,} | vparty parties: {len(vp):,} | "
          f"bridge speakers: {len(bridge):,}")

    con = sqlite3.connect(config.ACC_DB)
    con.executescript(SCHEMA)
    con.execute("DELETE FROM party_vars")
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, speaker, country, source_dataset, date, is_interjection, accuser_speaker "
        "FROM accusations" + (f" LIMIT {args.limit}" if args.limit else "")
    ).fetchall()

    stats = {"resolved": 0, "no_speaker": 0, "no_party": 0}
    batch = []
    for r in rows:
        # accuser identity: for interjections use the resolved interjector
        accuser = (r["accuser_speaker"] if (r["is_interjection"] and r["accuser_speaker"])
                   else r["speaker"])
        key = (accuser, r["country"], r["source_dataset"])
        parties = bridge.get(key)
        D = _d(r["date"])
        if not parties or not D:
            stats["no_speaker" if not parties else "no_party"] += 1
            batch.append((r["id"], None, "no_speaker" if not parties else "no_party",
                          0, 0, *[None]*N_NULL_TAIL))
            continue
        p = party_at_year(parties, int(D[:4]))
        if not p or p.get("partyfacts_id") is None:
            stats["no_party"] += 1
            batch.append((r["id"], None, "no_party", 0, 0, *[None]*N_NULL_TAIL))
            continue

        pfid = int(p["partyfacts_id"])
        pg_id = xwalk.get(pfid)
        ev = election_vars(pg_id, D, elec) if pg_id else (None, None, None, None)
        cv = cabinet_vars(pg_id, D, cab_by_country, party2country) if pg_id else (None, None, None)
        vv = vparty_vars(pfid, D, vp)
        stats["resolved"] += 1
        batch.append((
            r["id"], pfid, "resolved",
            1 if pg_id else 0, 1 if pfid in vp else 0,
            ev[0], ev[1], ev[2], ev[3],
            cv[0], cv[1], cv[2],
            *[vv[m] for m in VPARTY_MEASURES],
        ))

    n_cols = 5 + N_NULL_TAIL
    con.executemany(
        f"INSERT INTO party_vars VALUES ({','.join('?' * n_cols)})", batch)
    con.commit()
    con.close()

    print(f"\nDone.  {len(rows):,} accusations")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
