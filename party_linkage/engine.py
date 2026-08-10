"""
engine.py
=========
Reusable per-(speaker, date) variable engine for the analysis-ready datasets.

Combines, for any corpus speaker on any date:
  - individual-level enrichment (Group A: gender, birth_year -> age, birth_place;
    Group C: highest_isced; Group D: career sectors)
  - party at that date (Group B timeline)
  - party-level data (ParlGov electoral/cabinet + V-Party ideology/populism)

Both dataset builders call Engine.vars_for(speaker, country, dataset, date) and
prefix the returned dict with 'accuser_' / 'target_' / '' as needed.
"""

import json
import sqlite3

import pandas as pd

import config
from build_party_vars import (
    _d, load_crosswalk_parlgov, load_elections, load_cabinets, load_vparty,
    party_at_year, election_vars, cabinet_vars, vparty_vars, VPARTY_MEASURES,
)

# Output variable keys (the flat schema returned by vars_for)
VAR_KEYS = [
    "match", "speaker_id",
    "gender", "birth_year", "age", "birth_place", "highest_isced", "career_sectors",
    "partyfacts_id", "party_name",
    "vote_share_last", "vote_share_avg", "vote_share_avg_last3", "vote_share_delta",
    "in_cabinet", "is_pm_party", "years_since_government",
] + VPARTY_MEASURES


def _load_speakers_full():
    """(speaker, country, source_dataset) -> record with speaker_id + individual
    enrichment + party timeline."""
    sn = pd.read_csv(config.SPEAKER_NAMES, dtype=str).fillna("")
    con = sqlite3.connect(config.ENRICH_DB)
    sp = pd.read_sql_query(
        "SELECT speaker_id, name_cleaned, country, source_dataset FROM speakers", con)
    name2id = {(r.name_cleaned, r.country, r.source_dataset): r.speaker_id
               for r in sp.itertuples(index=False)}
    ann = pd.read_sql_query(
        "SELECT speaker_id, group_name, annotation_json FROM speaker_annotations "
        "WHERE status='success'", con)
    con.close()

    byid = {}
    for r in ann.itertuples(index=False):
        try:
            j = json.loads(r.annotation_json) if r.annotation_json else {}
        except Exception:
            j = {}
        byid.setdefault(r.speaker_id, {})[r.group_name] = j or {}

    out = {}
    for r in sn.itertuples(index=False):
        sid = name2id.get((r.name_cleaned, r.country, r.source_dataset))
        if not sid:
            continue
        a = byid.get(sid, {})
        A, B, C, D = a.get("A", {}), a.get("B", {}), a.get("C", {}), a.get("D", {})
        sectors = D.get("sectors")
        if isinstance(sectors, list):
            sectors = ",".join(str(x) for x in sectors)
        out[(r.speaker, r.country, r.source_dataset)] = {
            "speaker_id":    sid,
            "gender":        A.get("gender"),
            "birth_year":    A.get("birth_year"),
            "birth_place":   A.get("birth_place"),
            "highest_isced": C.get("highest_isced"),
            "career_sectors": sectors,
            "parties":       B.get("parties", []),
        }
    return out


class Engine:
    def __init__(self):
        print("  loading speaker enrichment + party timelines ...")
        self.speakers = _load_speakers_full()
        print("  loading ParlGov + V-Party ...")
        self.xwalk = load_crosswalk_parlgov()
        self.elec = load_elections()
        self.cab_by_country, self.party2country = load_cabinets()
        self.vp = load_vparty()
        self._cache = {}
        print(f"  engine ready: {len(self.speakers):,} corpus speakers")

    def vars_for(self, speaker, country, dataset, date):
        """Return the flat variable dict for one speaker on one date."""
        key = (speaker, country, dataset, date)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = {k: None for k in VAR_KEYS}
        rec = self.speakers.get((speaker, country, dataset))
        D = _d(date)
        if rec is None:
            out["match"] = "no_speaker"
            self._cache[key] = out
            return out

        out["speaker_id"]    = rec["speaker_id"]
        out["gender"]        = rec["gender"]
        out["birth_year"]    = rec["birth_year"]
        out["birth_place"]   = rec["birth_place"]
        out["highest_isced"] = rec["highest_isced"]
        out["career_sectors"] = rec["career_sectors"]
        try:
            if rec["birth_year"] and D:
                out["age"] = int(D[:4]) - int(rec["birth_year"])
        except (ValueError, TypeError):
            pass

        yr = int(D[:4]) if D else None
        p = party_at_year(rec["parties"], yr) if yr else None
        if p and p.get("partyfacts_id") is not None:
            pfid = int(p["partyfacts_id"])
            out["partyfacts_id"] = pfid
            out["party_name"] = p.get("partyfacts_name")
            pg = self.xwalk.get(pfid)
            ev = election_vars(pg, D, self.elec) if pg else (None, None, None, None)
            cv = cabinet_vars(pg, D, self.cab_by_country, self.party2country) if pg else (None, None, None)
            vv = vparty_vars(pfid, D, self.vp)
            (out["vote_share_last"], out["vote_share_avg"],
             out["vote_share_avg_last3"], out["vote_share_delta"]) = ev
            out["in_cabinet"], out["is_pm_party"], out["years_since_government"] = cv
            out.update(vv)
            out["match"] = "resolved"
        else:
            out["match"] = "no_party"

        self._cache[key] = out
        return out
