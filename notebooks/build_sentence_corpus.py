"""
build_sentence_corpus.py

Splits all parliamentary speeches to sentence level and produces one unified CSV.
Each row is one sentence with back-pointers to its source.

Output columns:
    sentence           - English sentence text
    sentence_idx       - position within the original speech (0-based)
    date               - speech date
    speaker            - speaker name or ID
    country            - ISO 2-letter country code (ParlaMint) or ISO3 (ParlSpeech/Italy)
    source_file        - filename the speech came from
    source_speech_id   - identifier for the speech within the source file
                         (utterance xml:id for ParlaMint, speechnumber/row index for ParlSpeech)
    source_dataset     - specific corpus name (e.g. "ParlaMint-DE", "Bundestag", "italy")
    source_dataset_type - "parlamint", "parlspeech", or "parlspeech_italy"
"""

import os
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from glob import glob

import nltk
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — adjust paths as needed
# ---------------------------------------------------------------------------
PARLAMINT_DIR = "/home/tom/data/parlamint/raw/parlamint"
PARLSPEECH_TRANSLATED_DIR = "/home/tom/projects/corpora"   # translated_*.csv files
ITALY_TRANSLATED_FILE = "/home/tom/projects/corpora/translated_italy.csv"
OUTPUT_FILE = "/home/tom/data/sentence_corpus.csv"
CHUNK_SIZE = 10_000  # flush to disk every N sentences

# ParlaMint countries to skip (native English — include them but note the source)
# Add ISO-2 codes here if you want to exclude any country
PARLAMINT_SKIP = set()

# ParlSpeech dataset name → ISO3 country code mapping (for source_dataset label)
PARLSPEECH_DATASETS = {
    "Bundestag":        "DEU",
    "Congreso":         "ESP",
    "Folketing":        "DNK",
    "PSP":              "CZE",
    "TweedeKamer":      "NLD",
    "Corp_Riksdagen_V2": "SWE",
}

# For ParlSpeech datasets that overlap with ParlaMint, only include rows
# strictly BEFORE this year (ParlaMint is preferred for the overlapping period).
# None = no ParlaMint equivalent for this country, include all years.
PARLSPEECH_PARLAMINT_CUTOFFS = {
    "Bundestag":         2021,   # ParlaMint-DE covers 2021–2022
    "Congreso":          2015,   # ParlaMint-ES covers 2015+
    "Folketing":         2014,   # ParlaMint-DK covers 2014+
    "PSP":               2014,   # ParlaMint-CZ covers 2014+
    "TweedeKamer":       2014,   # ParlaMint-NL covers 2014+
    "Corp_Riksdagen_V2": None,   # No ParlaMint-SE in dataset — include all
}
# ---------------------------------------------------------------------------

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def ensure_nltk():
    for resource in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1])


def split_sentences(text: str) -> list[str]:
    """Split English text into sentences with NLTK."""
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        return [text.strip()] if text.strip() else []


def write_rows(writer, rows: list[dict]):
    for row in rows:
        writer.writerow(row)


# ---------------------------------------------------------------------------
# ParlaMint (TEI XML)
# ---------------------------------------------------------------------------

def extract_date_parlamint(root) -> str | None:
    """Extract date string from ParlaMint TEI header."""
    paths = [
        ".//tei:teiHeader//tei:sourceDesc//tei:bibl//tei:date",
        ".//tei:teiHeader//tei:profileDesc//tei:settingDesc//tei:setting//tei:date",
    ]
    for path in paths:
        elem = root.find(path, TEI_NS)
        if elem is not None and "when" in elem.attrib:
            return elem.attrib["when"]
    return None


def iter_parlamint_rows(parlamint_dir: str):
    """
    Yield sentence-level dicts for all ParlaMint countries.
    Reads from ParlaMint-XX-en.ana folders (English-translated versions).
    """
    country_dirs = sorted([
        d for d in os.listdir(parlamint_dir)
        if d.startswith("ParlaMint-") and d.endswith("-en.ana")
        and os.path.isdir(os.path.join(parlamint_dir, d))
    ])

    if not country_dirs:
        # Fallback: any ParlaMint-XX folder (for native-English corpora)
        country_dirs = sorted([
            d for d in os.listdir(parlamint_dir)
            if d.startswith("ParlaMint-") and os.path.isdir(os.path.join(parlamint_dir, d))
        ])

    for country_folder in tqdm(country_dirs, desc="ParlaMint countries"):
        # Country code: ParlaMint-GB-en.ana → GB
        parts = country_folder.split("-")
        country_code = parts[1] if len(parts) >= 2 else country_folder

        if country_code in PARLAMINT_SKIP:
            continue

        dataset_name = f"ParlaMint-{country_code}"
        folder_path = os.path.join(parlamint_dir, country_folder)

        # Find the .TEI.ana subfolder
        tei_subdirs = [
            os.path.join(folder_path, d)
            for d in os.listdir(folder_path)
            if d.endswith(".TEI.ana") and os.path.isdir(os.path.join(folder_path, d))
        ]
        if not tei_subdirs:
            print(f"  ⚠️  No .TEI.ana subfolder for {country_folder}, skipping.")
            continue

        ana_dir = tei_subdirs[0]

        # Collect XML files (flat or in year subdirs)
        xml_files = sorted(glob(os.path.join(ana_dir, "**", "*.xml"), recursive=True))

        country_sentences = 0
        for xml_file in tqdm(xml_files, desc=f"  {country_code}", leave=False, unit="file"):
            filename = os.path.basename(xml_file)
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                date_str = extract_date_parlamint(root)

                utterances = root.findall(".//tei:u", TEI_NS)
                for u in utterances:
                    uid = u.attrib.get("{http://www.w3.org/XML/1998/namespace}id", "")
                    speaker = u.attrib.get("who", "").lstrip("#")
                    text = " ".join(u.itertext()).strip()
                    if not text:
                        continue

                    sentences = split_sentences(text)
                    country_sentences += len(sentences)
                    for idx, sentence in enumerate(sentences):
                        yield {
                            "sentence": sentence,
                            "sentence_idx": idx,
                            "date": date_str,
                            "speaker": speaker,
                            "country": country_code,
                            "source_file": filename,
                            "source_speech_id": uid,
                            "source_dataset": dataset_name,
                            "source_dataset_type": "parlamint",
                        }
            except Exception as e:
                tqdm.write(f"  ⚠️  Error parsing {xml_file}: {e}")
                continue

        tqdm.write(f"  ✓ {dataset_name}: {country_sentences:,} sentences from {len(xml_files):,} files")


# ---------------------------------------------------------------------------
# ParlSpeech (translated CSVs)
# ---------------------------------------------------------------------------

def iter_parlspeech_rows(translated_dir: str):
    """
    Yield sentence-level dicts for all ParlSpeech translated CSV files.
    Reads from translated_<DatasetName>.csv files.
    """
    for dataset_name, iso3 in PARLSPEECH_DATASETS.items():
        csv_path = os.path.join(translated_dir, f"translated_{dataset_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  ⚠️  Not found, skipping: {csv_path}")
            continue

        cutoff_year = PARLSPEECH_PARLAMINT_CUTOFFS.get(dataset_name)

        print(f"  Processing ParlSpeech: {dataset_name}"
              + (f" (rows before {cutoff_year} only — rest covered by ParlaMint)" if cutoff_year else ""))
        df = pd.read_csv(csv_path, low_memory=False)

        if "en_translation" not in df.columns:
            print(f"  ⚠️  No en_translation column in {csv_path}, skipping.")
            continue

        if cutoff_year is not None:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            before = len(df)
            df = df[df["date"].dt.year < cutoff_year]
            print(f"    Kept {len(df):,} / {before:,} rows (year < {cutoff_year})")

        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=dataset_name, leave=False):
            text = row.get("en_translation")
            if pd.isna(text) or not str(text).strip():
                continue

            date = str(row.get("date", "")) if not pd.isna(row.get("date", float("nan"))) else ""
            speaker = str(row.get("speaker", "")) if not pd.isna(row.get("speaker", float("nan"))) else ""
            speech_id = str(row.get("speechnumber", row_idx))

            sentences = split_sentences(str(text))
            for idx, sentence in enumerate(sentences):
                yield {
                    "sentence": sentence,
                    "sentence_idx": idx,
                    "date": date,
                    "speaker": speaker,
                    "country": iso3,
                    "source_file": f"translated_{dataset_name}.csv",
                    "source_speech_id": speech_id,
                    "source_dataset": dataset_name,
                    "source_dataset_type": "parlspeech",
                }


# ---------------------------------------------------------------------------
# Italian dataset
# ---------------------------------------------------------------------------

def iter_italy_rows(italy_file: str):
    """
    Yield sentence-level dicts for the Italian translated CSV.
    """
    if not os.path.exists(italy_file):
        print(f"  ⚠️  Italian file not found: {italy_file}, skipping.")
        return

    print("  Processing Italian dataset")
    df = pd.read_csv(italy_file, low_memory=False)

    if "en_translation" not in df.columns:
        print(f"  ⚠️  No en_translation column in {italy_file}, skipping.")
        return

    filename = os.path.basename(italy_file)

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="italy", leave=False):
        text = row.get("en_translation")
        if pd.isna(text) or not str(text).strip():
            continue

        date = str(row.get("date", "")) if not pd.isna(row.get("date", float("nan"))) else ""
        speaker = str(row.get("speaker", "")) if not pd.isna(row.get("speaker", float("nan"))) else ""
        speech_id = str(row_idx)

        sentences = split_sentences(str(text))
        for idx, sentence in enumerate(sentences):
            yield {
                "sentence": sentence,
                "sentence_idx": idx,
                "date": date,
                "speaker": speaker,
                "country": "ITA",
                "source_file": filename,
                "source_speech_id": speech_id,
                "source_dataset": "italy",
                "source_dataset_type": "parlspeech_italy",
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "sentence",
    "sentence_idx",
    "date",
    "speaker",
    "country",
    "source_file",
    "source_speech_id",
    "source_dataset",
    "source_dataset_type",
]


def main():
    ensure_nltk()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total_sentences = 0
    buffer = []

    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"Building sentence corpus")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    last_report = start
    report_interval = 30  # seconds between progress lines

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        def flush(buf):
            nonlocal last_report
            write_rows(writer, buf)
            f.flush()
            now = datetime.now()
            if (now - last_report).total_seconds() >= report_interval:
                elapsed = (now - start).total_seconds()
                rate = total_sentences / elapsed if elapsed > 0 else 0
                print(f"  [{now.strftime('%H:%M:%S')}] {total_sentences:,} sentences written"
                      f"  ({rate:,.0f} sent/s,  {elapsed/60:.1f} min elapsed)")
                last_report = now
            return []

        # --- ParlaMint ---
        print("Processing ParlaMint...")
        for row in iter_parlamint_rows(PARLAMINT_DIR):
            buffer.append(row)
            total_sentences += 1
            if len(buffer) >= CHUNK_SIZE:
                buffer = flush(buffer)
        buffer = flush(buffer)
        print(f"  ParlaMint done. Running total: {total_sentences:,} sentences\n")

        # --- ParlSpeech ---
        print("Processing ParlSpeech...")
        for row in iter_parlspeech_rows(PARLSPEECH_TRANSLATED_DIR):
            buffer.append(row)
            total_sentences += 1
            if len(buffer) >= CHUNK_SIZE:
                buffer = flush(buffer)
        buffer = flush(buffer)
        print(f"  ParlSpeech done. Running total: {total_sentences:,} sentences\n")

        # --- Italy ---
        # TODO: enable once translation is complete
        # print("Processing Italian dataset...")
        # for row in iter_italy_rows(ITALY_TRANSLATED_FILE):
        #     buffer.append(row)
        #     total_sentences += 1
        #     if len(buffer) >= CHUNK_SIZE:
        #         buffer = flush(buffer)
        # buffer = flush(buffer)
        # print(f"  Italy done. Running total: {total_sentences:,} sentences\n")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"{'='*70}")
    print(f"Done. {total_sentences:,} sentences written to {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
