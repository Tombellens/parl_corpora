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
    source_dataset     - specific corpus name (e.g. "ParlaMint-DE", "Bundestag", "italy",
                         "hein-daily", "congressional-record")
    source_dataset_type - "parlamint", "parlspeech", "parlspeech_italy",
                          "gentzkow", "congressional-record",
                          "australian-hansard", or "hansard"
"""

import os
import csv
import json
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

# US datasets
GENTZKOW_DIR = "/home/tom/data/us/gentzkow_et_al/hein-daily"
CONGRESSIONAL_RECORD_DIR = "/home/tom/data/us/congressional-record/output"
# Gentzkow hein-daily covers up to 114th Congress (ends Jan 2017).
# Congressional Record JSON data starts from 2016.
# Use Gentzkow for < cutoff, Congressional Record for >= cutoff.
US_CR_CUTOFF_YEAR = 2016

# Australian Hansard (Katz et al. 2023, Zenodo)
AUSTRALIA_HANSARD_FILE = "/home/tom/data/australia/hansard_corpus_1998_to_2022.csv"

# Canadian Hansard (LiPaD) — PostgreSQL database
# Setup: sudo -u postgres psql -c "CREATE USER tom;"
#        sudo -u postgres psql -c "GRANT CONNECT ON DATABASE hansard TO tom;"
#        sudo -u postgres psql hansard -c "GRANT SELECT ON ALL TABLES IN SCHEMA public TO tom;"
#        pip install psycopg2-binary
HANSARD_DB_NAME = "hansard"
HANSARD_DB_USER = "tom"       # Unix peer auth — must match OS username
HANSARD_START_YEAR = 1990     # LiPaD covers 1901–2019; we start from 1990
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
                    text = " ".join(u.itertext())
                    text = " ".join(text.split())  # collapse all whitespace incl. newlines
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
# US: Gentzkow et al. hein-daily
# ---------------------------------------------------------------------------

def iter_gentzkow_rows(hein_daily_dir: str):
    """
    Yield sentence-level dicts for the Gentzkow et al. hein-daily dataset.

    File layout (per congress, e.g. 114):
        speeches_114.txt  — pipe-delimited: speech_id|speech
        descr_114.txt     — pipe-delimited: speech_id|chamber|date(YYYYMMDD)|...|
                            first_name|last_name|speaker|...

    Only speeches with date < US_CR_CUTOFF_YEAR are included
    (Congressional Record covers the rest).
    """
    import re

    speech_files = sorted(glob(os.path.join(hein_daily_dir, "speeches_*.txt")))
    if not speech_files:
        print(f"  ⚠️  No Gentzkow speech files found in {hein_daily_dir}")
        return

    for speech_file in tqdm(speech_files, desc="Gentzkow congresses"):
        m = re.search(r'speeches_(\d+)\.txt$', os.path.basename(speech_file))
        if not m:
            continue
        congress = m.group(1)
        descr_file = os.path.join(hein_daily_dir, f"descr_{congress}.txt")

        if not os.path.exists(descr_file):
            tqdm.write(f"  ⚠️  No descr file for congress {congress}, skipping.")
            continue

        # Load descr metadata
        try:
            descr_df = pd.read_csv(descr_file, sep='|', dtype=str, low_memory=False)
        except Exception as e:
            tqdm.write(f"  ⚠️  Error reading {descr_file}: {e}, skipping.")
            continue

        required_cols = {'speech_id', 'date'}
        if not required_cols.issubset(descr_df.columns):
            tqdm.write(f"  ⚠️  Missing columns in {descr_file} "
                       f"(found: {list(descr_df.columns)}), skipping.")
            continue

        # Parse dates and apply cutoff
        descr_df['date_parsed'] = pd.to_datetime(
            descr_df['date'], format='%Y%m%d', errors='coerce')
        before = len(descr_df)
        descr_df = descr_df[descr_df['date_parsed'].dt.year < US_CR_CUTOFF_YEAR]
        if descr_df.empty:
            tqdm.write(f"  Skipping congress {congress}: "
                       f"all {before:,} speeches at or after cutoff year {US_CR_CUTOFF_YEAR}.")
            continue

        # Format date as YYYY-MM-DD
        descr_df['date_str'] = (
            descr_df['date_parsed'].dt.strftime('%Y-%m-%d').fillna(''))

        # Build speaker name: prefer first_name + last_name, fall back to speaker column
        _UNKNOWN = {'Unknown', 'unknown', 'nan', 'None', '', 'NaN'}
        first = descr_df.get('first_name', pd.Series('', index=descr_df.index)).fillna('')
        last  = descr_df.get('last_name',  pd.Series('', index=descr_df.index)).fillna('')
        known = ~first.isin(_UNKNOWN) & ~last.isin(_UNKNOWN)
        descr_df['speaker_name'] = ''
        descr_df.loc[known, 'speaker_name'] = first[known] + ' ' + last[known]
        if 'speaker' in descr_df.columns:
            fallback_mask = ~known
            fb = descr_df.loc[fallback_mask, 'speaker'].fillna('').str.strip()
            fb = fb.where(~fb.isin(_UNKNOWN), '')
            descr_df.loc[fallback_mask, 'speaker_name'] = fb

        # Build lookup dict: speech_id -> {date_str, speaker_name}
        meta_lookup = (
            descr_df.set_index('speech_id')[['date_str', 'speaker_name']]
            .to_dict('index')
        )
        valid_ids = set(meta_lookup.keys())
        filename = os.path.basename(speech_file)
        congress_sentences = 0

        # Stream speeches file — split on first '|' only (speech text may contain '|')
        with open(speech_file, 'r', encoding='utf-8', errors='replace') as f:
            next(f)  # skip header
            for line in f:
                line = line.rstrip('\n\r')
                if '|' not in line:
                    continue
                speech_id, speech_text = line.split('|', 1)
                if speech_id not in valid_ids:
                    continue

                speech_text = ' '.join(speech_text.split())
                if not speech_text:
                    continue

                meta = meta_lookup[speech_id]
                sentences = split_sentences(speech_text)
                congress_sentences += len(sentences)
                for idx, sentence in enumerate(sentences):
                    yield {
                        'sentence':          sentence,
                        'sentence_idx':      idx,
                        'date':              meta['date_str'],
                        'speaker':           meta['speaker_name'],
                        'country':           'USA',
                        'source_file':       filename,
                        'source_speech_id':  speech_id,
                        'source_dataset':    'hein-daily',
                        'source_dataset_type': 'gentzkow',
                    }

        tqdm.write(f"  ✓ Congress {congress}: {congress_sentences:,} sentences "
                   f"(from {len(valid_ids):,} / {before:,} speeches before {US_CR_CUTOFF_YEAR})")


# ---------------------------------------------------------------------------
# US: Congressional Record (JSON)
# ---------------------------------------------------------------------------

def iter_congressional_record_rows(cr_output_dir: str):
    """
    Yield sentence-level dicts for the Congressional Record JSON dataset.

    File layout:
        output/{year}/CREC-{YYYY-MM-DD}/json/CREC-*.json

    Each JSON has:
        header.year / header.month / header.day  — date
        content[].kind     — "speech" or "linebreak"
        content[].text     — speech text
        content[].speaker  — speaker name string (may be "None")
        content[].itemno   — position within file

    Only years >= US_CR_CUTOFF_YEAR are included (Gentzkow covers the rest).
    """
    if not os.path.isdir(cr_output_dir):
        print(f"  ⚠️  Congressional Record output dir not found: {cr_output_dir}")
        return

    years = sorted([
        y for y in os.listdir(cr_output_dir)
        if y.isdigit() and int(y) >= US_CR_CUTOFF_YEAR
        and os.path.isdir(os.path.join(cr_output_dir, y))
    ])
    if not years:
        print(f"  ⚠️  No years >= {US_CR_CUTOFF_YEAR} found in {cr_output_dir}")
        return

    for year in tqdm(years, desc="Congressional Record years"):
        year_dir = os.path.join(cr_output_dir, year)
        day_dirs = sorted([
            d for d in os.listdir(year_dir)
            if d.startswith('CREC-') and os.path.isdir(os.path.join(year_dir, d))
        ])

        year_sentences = 0
        for day_dir in tqdm(day_dirs, desc=f"  {year}", leave=False):
            json_dir = os.path.join(year_dir, day_dir, 'json')
            if not os.path.isdir(json_dir):
                continue

            for json_file in sorted(glob(os.path.join(json_dir, '*.json'))):
                filename = os.path.basename(json_file)
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    tqdm.write(f"  ⚠️  Error reading {json_file}: {e}")
                    continue

                # Build date string from header
                header = data.get('header', {})
                date_str = ''
                try:
                    month_num = datetime.strptime(header.get('month', ''), '%B').month
                    date_str = (f"{header['year']}-{month_num:02d}"
                                f"-{int(header['day']):02d}")
                except (ValueError, KeyError):
                    pass

                doc_id = data.get('id', filename.replace('.json', ''))

                for item in data.get('content', []):
                    if item.get('kind') != 'speech':
                        continue

                    text = item.get('text', '')
                    text = ' '.join(text.split())  # collapse all whitespace
                    if not text:
                        continue

                    speaker = item.get('speaker', '') or ''
                    if speaker in ('None', 'none', 'nan'):
                        speaker = ''

                    itemno = item.get('itemno', 0)
                    speech_id = f"{doc_id}-item{itemno}"

                    sentences = split_sentences(text)
                    year_sentences += len(sentences)
                    for idx, sentence in enumerate(sentences):
                        yield {
                            'sentence':          sentence,
                            'sentence_idx':      idx,
                            'date':              date_str,
                            'speaker':           speaker,
                            'country':           'USA',
                            'source_file':       filename,
                            'source_speech_id':  speech_id,
                            'source_dataset':    'congressional-record',
                            'source_dataset_type': 'congressional-record',
                        }

        tqdm.write(f"  ✓ Year {year}: {year_sentences:,} sentences "
                   f"from {len(day_dirs):,} daily issues")


# ---------------------------------------------------------------------------
# Australia: Katz et al. 2023 Hansard CSV
# ---------------------------------------------------------------------------

def iter_australia_rows(hansard_file: str):
    """
    Yield sentence-level dicts for the Australian Hansard dataset.

    Source: Katz et al. (2023) 'Digitization of the Australian Parliamentary
    Debates, 1998-2022', Scientific Data / Zenodo.
    File: hansard_corpus_1998_to_2022.csv (single large CSV)

    Key columns:
        date      — YYYY-MM-DD
        name      — speaker name (may be procedural label e.g. 'Business start')
        body      — speech text
        uniqueID  — unique speech identifier (may be NA)
        speech_no — fallback speech number (may be NA)
    """
    if not os.path.exists(hansard_file):
        print(f"  ⚠️  Australian Hansard file not found: {hansard_file}")
        return

    filename = os.path.basename(hansard_file)
    print(f"  Processing Australian Hansard: {hansard_file}")

    total_sentences = 0
    chunk_num = 0

    for chunk in pd.read_csv(
        hansard_file,
        usecols=["date", "name", "body", "uniqueID", "speech_no"],
        dtype=str,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_num += 1

        # Drop rows with no speech body
        chunk = chunk[chunk["body"].notna() & (chunk["body"].str.strip() != "")]

        for _, row in tqdm(
            chunk.iterrows(),
            total=len(chunk),
            desc=f"  Australian Hansard chunk {chunk_num}",
            leave=False,
        ):
            text = " ".join(str(row["body"]).split())  # collapse whitespace
            if not text:
                continue

            date_str   = str(row["date"]) if pd.notna(row["date"]) else ""
            speaker    = str(row["name"]) if pd.notna(row["name"]) else ""

            # Prefer uniqueID, fall back to speech_no, then row index
            uid = row.get("uniqueID", "")
            if pd.isna(uid) or str(uid).strip() in ("", "NA", "nan"):
                uid = row.get("speech_no", "")
            if pd.isna(uid) or str(uid).strip() in ("", "NA", "nan"):
                uid = str(_)
            speech_id = str(uid).strip()

            sentences = split_sentences(text)
            total_sentences += len(sentences)
            for idx, sentence in enumerate(sentences):
                yield {
                    "sentence":           sentence,
                    "sentence_idx":       idx,
                    "date":               date_str,
                    "speaker":            speaker,
                    "country":            "AUS",
                    "source_file":        filename,
                    "source_speech_id":   speech_id,
                    "source_dataset":     "australian-hansard",
                    "source_dataset_type": "australian-hansard",
                }

    tqdm.write(f"  ✓ Australian Hansard: {total_sentences:,} sentences")


# ---------------------------------------------------------------------------
# Canada: LiPaD / basehansard (PostgreSQL)
# ---------------------------------------------------------------------------

def iter_hansard_rows(
    dbname: str = HANSARD_DB_NAME,
    user: str = HANSARD_DB_USER,
    start_year: int = HANSARD_START_YEAR,
):
    """
    Yield sentence-level dicts for the Canadian Hansard (LiPaD) dataset.

    Streams directly from PostgreSQL using a server-side named cursor —
    never loads the full table into memory.

    Requires psycopg2:  pip install psycopg2-binary

    PostgreSQL setup (one-time):
        sudo -u postgres psql -c "CREATE USER tom;"
        sudo -u postgres psql -c "GRANT CONNECT ON DATABASE hansard TO tom;"
        sudo -u postgres psql hansard -c \
            "GRANT SELECT ON ALL TABLES IN SCHEMA public TO tom;"

    Key columns in dilipadsite_basehansard:
        basepk      — integer primary key  → source_speech_id
        speechdate  — date (YYYY-MM-DD)    → date
        speakername — speaker name string  → speaker
        speechtext  — English speech text  → sentence corpus
    """
    try:
        import psycopg2
    except ImportError:
        print("  ⚠️  psycopg2 not installed. Run: pip install psycopg2-binary")
        return

    try:
        # Unix domain socket peer auth — no host/password needed when
        # OS username matches PostgreSQL username.
        conn = psycopg2.connect(dbname=dbname, user=user)
    except Exception as e:
        print(f"  ⚠️  Could not connect to PostgreSQL hansard DB: {e}")
        return

    start_date = f"{start_year}-01-01"

    try:
        # Count for progress bar
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM dilipadsite_basehansard "
                "WHERE speechdate >= %s "
                "  AND speechtext IS NOT NULL AND speechtext <> ''",
                (start_date,)
            )
            total_speeches = cur.fetchone()[0]

        print(f"  Found {total_speeches:,} speeches in hansard DB "
              f"from {start_year}+")

        # Server-side named cursor: fetches HANSARD_FETCH_SIZE rows at a time
        with conn.cursor("hansard_stream") as cur:
            cur.itersize = 10_000
            cur.execute(
                """
                SELECT basepk, speechdate, speakername, speechtext
                FROM dilipadsite_basehansard
                WHERE speechdate >= %s
                  AND speechtext IS NOT NULL
                  AND speechtext <> ''
                ORDER BY speechdate, basepk
                """,
                (start_date,)
            )

            total_sentences = 0
            for row in tqdm(cur, total=total_speeches, desc="Canadian Hansard"):
                basepk, speechdate, speakername, speechtext = row

                text = " ".join(str(speechtext).split())  # collapse whitespace
                if not text:
                    continue

                speaker  = str(speakername).strip() if speakername else ""
                date_str = str(speechdate) if speechdate else ""
                speech_id = str(basepk)

                sentences = split_sentences(text)
                total_sentences += len(sentences)
                for idx, sentence in enumerate(sentences):
                    yield {
                        "sentence":           sentence,
                        "sentence_idx":       idx,
                        "date":               date_str,
                        "speaker":            speaker,
                        "country":            "CAN",
                        "source_file":        "dilipadsite_basehansard",
                        "source_speech_id":   speech_id,
                        "source_dataset":     "lipad",
                        "source_dataset_type": "hansard",
                    }

        tqdm.write(f"  ✓ Canadian Hansard: {total_sentences:,} sentences "
                   f"from {total_speeches:,} speeches ({start_year}+)")

    finally:
        conn.close()


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

        # --- US: Gentzkow et al. hein-daily (pre-2016) ---
        # TODO: enable once US datasets are ready to include
        # print("Processing Gentzkow et al. (US, pre-2016)...")
        # for row in iter_gentzkow_rows(GENTZKOW_DIR):
        #     buffer.append(row)
        #     total_sentences += 1
        #     if len(buffer) >= CHUNK_SIZE:
        #         buffer = flush(buffer)
        # buffer = flush(buffer)
        # print(f"  Gentzkow done. Running total: {total_sentences:,} sentences\n")

        # --- US: Congressional Record (2016+) ---
        # TODO: enable once US datasets are ready to include
        # print("Processing Congressional Record (US, 2016+)...")
        # for row in iter_congressional_record_rows(CONGRESSIONAL_RECORD_DIR):
        #     buffer.append(row)
        #     total_sentences += 1
        #     if len(buffer) >= CHUNK_SIZE:
        #         buffer = flush(buffer)
        # buffer = flush(buffer)
        # print(f"  Congressional Record done. Running total: {total_sentences:,} sentences\n")

        # --- Australia: Katz et al. Hansard (1998–2022) ---
        # TODO: enable when ready
        # print("Processing Australian Hansard (1998–2022)...")
        # for row in iter_australia_rows(AUSTRALIA_HANSARD_FILE):
        #     buffer.append(row)
        #     total_sentences += 1
        #     if len(buffer) >= CHUNK_SIZE:
        #         buffer = flush(buffer)
        # buffer = flush(buffer)
        # print(f"  Australian Hansard done. Running total: {total_sentences:,} sentences\n")

        # --- Canada: LiPaD / basehansard (1990+) ---
        # TODO: enable once PostgreSQL user is configured and psycopg2 is installed
        # print("Processing Canadian Hansard (LiPaD, 1990+)...")
        # for row in iter_hansard_rows():
        #     buffer.append(row)
        #     total_sentences += 1
        #     if len(buffer) >= CHUNK_SIZE:
        #         buffer = flush(buffer)
        # buffer = flush(buffer)
        # print(f"  Canadian Hansard done. Running total: {total_sentences:,} sentences\n")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"{'='*70}")
    print(f"Done. {total_sentences:,} sentences written to {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
