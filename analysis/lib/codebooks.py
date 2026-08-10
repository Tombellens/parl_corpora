"""
codebooks.py — human-readable labels for the coded variables.

Single source of truth for value labels so every notebook renders the same
category names. Mirrors the enrichment prompts (Group C ISCED, Group D sectors)
and the target-detection Phase-1 type scheme.
"""

# Group C — highest_isced (ISCED 2011, tertiary only)
ISCED = {
    5: "Short-cycle tertiary",
    6: "Bachelor / equivalent",
    7: "Master / long first degree",
    8: "Doctorate",
}

# Group D — career sectors (first-digit high-level groups; national MP mandate excluded)
SECTOR = {
    1: "Central government executive",
    2: "Public administration",
    3: "Public sector orgs & industries",
    4: "Politics / political office (non-executive)",
    5: "Judiciary & oversight",
    6: "Private & third sector",
    7: "International organisations",
    8: "Other / none",
}

# Target detection — Phase-1 broad type
TARGET_TYPE = {
    "person": "Individual person",
    "government": "Government / executive",
    "administration": "Public administration",
    "political_party": "Political party",
    "public_institution": "Public institution",
    "foreign_country_or_govt": "Foreign country / government",
    "international_org": "International organisation",
    "media": "Media",
    "social_group": "Social group",
    "other": "Other",
    "unclear_or_none": "Unclear / none",
}

GENDER = {"male": "Male", "female": "Female"}

# ---------------------------------------------------------------------------
# V-Party scale polarity — CONFIRMED empirically, do not change without re-checking
# (00_overview/feasibility_checks.py --part a)
# ---------------------------------------------------------------------------
# left_right = V-Party v2pariglef, a standardised latent scale (~ -3.9 .. +4.2,
# mean 0.15), NOT the 0-10 survey scale. HIGHER = MORE RIGHT-WING, verified at
# both ends: lowest = Workers' Party BE (-3.94), Die Linke (-3.48), La France
# Insoumise (-3.46), KSCM (-3.35); highest = Vox (+4.17), Lega Nord (+2.68),
# Liberal Alliance (+2.74), Moderaterna (+2.43), ACT NZ (+2.39).
# => H4a (right accused more) expects a POSITIVE coefficient.
# => H4b (left accuses more) expects a NEGATIVE coefficient.
LEFT_RIGHT_HIGHER_IS = "right"

# CAVEAT for interpretation: v2pariglef appears to weight the economic dimension.
# Several culturally radical-right parties score only moderately right (PVV
# +1.66, FPOE < +1.66) while economically liberal parties score furthest right.
# Populism is therefore a genuinely separate dimension here — good for H2 vs H4
# identification, but do not read left_right as a radical-right proxy.


def assert_left_right_polarity(df, col="accuser_left_right", party_col="accuser_party_name"):
    """Fail loudly if the left_right scale is not oriented higher = right.

    Cheap insurance: call once per notebook that interprets a left_right sign.
    """
    import re
    left_pat = re.compile(r"communist|die linke|insoumise|socialis|left", re.I)
    right_pat = re.compile(r"\bvox\b|lega|moderat|conservative|progress party", re.I)
    sub = df.dropna(subset=[col, party_col])
    left = sub[sub[party_col].str.contains(left_pat)][col].mean()
    right = sub[sub[party_col].str.contains(right_pat)][col].mean()
    assert left < right, (
        f"left_right polarity looks REVERSED: left-named parties mean {left:.2f} "
        f"is not below right-named parties mean {right:.2f}. Re-run "
        f"00_overview/feasibility_checks.py --part a before interpreting signs.")
    return {"left_mean": left, "right_mean": right}

# Countries (ISO2) excluded from ALL analysis, mirroring the LieLines validation
# sampler (LieLines-Validation/sample_random.py DEFAULT_EXCLUDE_COUNTRIES).
# Iceland, Bosnia & Herzegovina, Greece, Latvia.
EXCLUDED_COUNTRIES = {"IS", "BA", "GR", "LV"}


def sectors_to_labels(sectors):
    """'1,4,6' or [1,4,6] -> ['Central government executive', ...]."""
    if sectors is None or sectors == "":
        return []
    if isinstance(sectors, str):
        parts = [p.strip() for p in sectors.split(",") if p.strip()]
    else:
        parts = list(sectors)
    out = []
    for p in parts:
        try:
            out.append(SECTOR[int(p)])
        except (ValueError, KeyError):
            continue
    return out
