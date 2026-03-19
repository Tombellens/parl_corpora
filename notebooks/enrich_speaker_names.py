"""
enrich_speaker_names.py
=======================
Adds a `name_cleaned` column to speaker_names.csv.

Strategy per dataset type:
    parlamint        - Look up xml:id in listPerson XML → "Forename Surname"
    parlspeech       - Speaker string is already "Firstname Lastname" → use as-is
    parlspeech_italy - ALL CAPS names → convert to Title Case
    australian-hansard, hansard, congressional-record
                     - Copy speaker string as-is for now; LLM cleanup pass later

Output: /home/tom/data/speaker_names.csv  (updated in-place, adds name_cleaned column)

Usage:
    python3 enrich_speaker_names.py
"""

import glob
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SPEAKER_NAMES_FILE = "/home/tom/data/speaker_names.csv"
PARLAMINT_DIR      = "/home/tom/data/parlamint/raw/parlamint"
TEI_NS             = "http://www.tei-c.org/ns/1.0"

# ---------------------------------------------------------------------------
# 1. ParlaMint: build xml:id → full name lookup from all listPerson XMLs
# ---------------------------------------------------------------------------

def build_parlamint_name_lookup(parlamint_dir: str) -> dict:
    """Return {xml:id: 'Forename Surname'} from all listPerson XML files,
    excluding the Samples directory."""
    ns      = {"tei": TEI_NS}
    lookup  = {}
    files   = [
        f for f in glob.glob(
            os.path.join(parlamint_dir, "**", "ParlaMint-*-listPerson.xml"),
            recursive=True,
        )
        if "Samples" not in f
    ]
    print(f"  Found {len(files)} listPerson XML files.")

    for fpath in files:
        try:
            tree    = ET.parse(fpath)
            persons = tree.findall(".//tei:person", ns)
            for person in persons:
                pid = person.get("{http://www.w3.org/XML/1998/namespace}id")
                if not pid:
                    continue
                forename = person.findtext("tei:persName/tei:forename", namespaces=ns, default="").strip()
                surname  = person.findtext("tei:persName/tei:surname",  namespaces=ns, default="").strip()
                full     = f"{forename} {surname}".strip()
                if full:
                    lookup[pid] = full
        except Exception as e:
            print(f"  ⚠️  Error parsing {fpath}: {e}")

    print(f"  Built lookup with {len(lookup):,} person IDs.")
    return lookup


# ---------------------------------------------------------------------------
# 2. Italy: normalise ALL CAPS to Title Case
# ---------------------------------------------------------------------------

def title_case_italian(name: str) -> str:
    """Convert 'ACHILLE OCCHETTO' → 'Achille Occhetto'.
    Handles particles and preserves short words."""
    return " ".join(
        word.capitalize() if len(word) > 1 else word
        for word in name.split()
    )


# ---------------------------------------------------------------------------
# 3. Main enrichment
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame, parlamint_lookup: dict) -> pd.DataFrame:
    name_cleaned = []

    for _, row in df.iterrows():
        dtype   = row["source_dataset_type"]
        speaker = str(row["speaker"]) if pd.notna(row["speaker"]) else ""

        if dtype == "parlamint":
            name = parlamint_lookup.get(speaker, "")

        elif dtype == "parlspeech":
            # Already "Firstname Lastname" — use as-is
            name = speaker.strip()

        elif dtype == "parlspeech_italy":
            # ALL CAPS → Title Case
            name = title_case_italian(speaker)

        else:
            # australian-hansard, hansard, congressional-record:
            # copy raw value; LLM cleanup pass later
            name = speaker.strip()

        name_cleaned.append(name if name else None)

    df = df.copy()
    df.insert(df.columns.get_loc("speaker") + 1, "name_cleaned", name_cleaned)
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {SPEAKER_NAMES_FILE}...")
    df = pd.read_csv(SPEAKER_NAMES_FILE, dtype=str)
    print(f"  {len(df):,} rows loaded.")

    print("\nBuilding ParlaMint name lookup...")
    parlamint_lookup = build_parlamint_name_lookup(PARLAMINT_DIR)

    print("\nEnriching names...")
    df = enrich(df, parlamint_lookup)

    # Summary
    total      = len(df)
    filled     = df["name_cleaned"].notna() & (df["name_cleaned"].str.strip() != "")
    print(f"\nname_cleaned filled: {filled.sum():,} / {total:,} "
          f"({filled.sum() / total * 100:.1f}%)")
    print("\nCoverage by dataset type:")
    print(
        df.groupby("source_dataset_type").apply(
            lambda g: f"{(g['name_cleaned'].notna() & (g['name_cleaned'].str.strip() != '')).sum()}"
                      f" / {len(g)}"
        ).to_string()
    )

    print(f"\nSample output:")
    print(df[filled].groupby("source_dataset_type").head(2)[
        ["source_dataset_type", "speaker", "name_cleaned"]
    ].to_string(index=False))

    print(f"\nSaving to {SPEAKER_NAMES_FILE}...")
    df.to_csv(SPEAKER_NAMES_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
