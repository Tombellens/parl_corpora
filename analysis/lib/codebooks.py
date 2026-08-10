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
