"""
classify_speaker_names.py
=========================
Uses a local LLM (via LM Studio's OpenAI-compatible API) to classify each
speaker name in speaker_names.csv as a real person or a non-person
(procedural label, committee name, title, etc.).

Adds an `is_person` column (True/False) and an `is_person_confidence` column
(high/low) to speaker_names.csv.

Skips rows already classified (safe to re-run after interruption).

Usage:
    python3 classify_speaker_names.py

Config:
    MODEL        - LM Studio model ID
    BATCH_SIZE   - names per API call (keep ~50 for reasoning models)
    MAX_TOKENS   - must be high enough for reasoning + JSON output
    BASE_URL     - LM Studio API endpoint
"""

import json
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SPEAKER_NAMES_FILE = "/home/tom/data/speaker_names.csv"
MODEL              = "openai/gpt-oss-20b"
BASE_URL           = "http://localhost:1234/v1"
BATCH_SIZE         = 50
MAX_TOKENS         = 2000
SAVE_EVERY         = 10   # save progress every N batches

SYSTEM_PROMPT = """You are a data cleaning assistant for a parliamentary speech corpus.

Your task: given a list of name strings from parliamentary records, classify each
as either a real person (an actual human being, e.g. a politician) or a non-person
(a procedural label, role title, committee name, place name, or any other non-person entry).

Examples of PERSONS: "Tony Abbott", "ACHILLE OCCHETTO", "Miss GONZALEZ-COLON",
  "Bruno Retailleau", "Adam Rykala", "PAD_00024" (ParlaMint speaker IDs are always persons)

Examples of NON-PERSONS: "ACTING SPEAKER", "The SPEAKER", "A government member",
  "DEPUTY PRIME MINISTER", "@Acting Speaker", "A Dynamic Economy for the 21st Century",
  "Business start", "The CLERK", "nan", "None"

Rules:
- ParlaMint IDs (format: letters + underscore + digits, e.g. PAD_00024, PA-100799) → always PERSON
- Strings that are clearly a role/title with no name → NON-PERSON
- Names with titles (Miss, Mr, Dr, Senator) but containing a real surname → PERSON
- Names with state/constituency suffixes (e.g. "COLLINS of Michigan") → PERSON
- Empty strings, "nan", "None" → NON-PERSON

Respond ONLY with a JSON array, one entry per input name, in the same order.
Each entry: {"name": "<original>", "is_person": true/false, "confidence": "high"/"low"}

Do not include any explanation or text outside the JSON array."""


def classify_batch(client: OpenAI, names: list[str]) -> list[dict]:
    """Send a batch of names to the LLM and return classification results."""
    numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(names))
    prompt   = f"Classify these {len(names)} name strings:\n\n{numbered}"

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0,
            )
            content = response.choices[0].message.content.strip()

            # Extract JSON array from response (handles markdown code blocks)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            results = json.loads(content)
            if len(results) == len(names):
                return results
            else:
                print(f"  ⚠️  Got {len(results)} results for {len(names)} names, retrying...")

        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error (attempt {attempt+1}): {e}")
            print(f"      Raw content: {content[:200]}")
        except Exception as e:
            print(f"  ⚠️  API error (attempt {attempt+1}): {e}")
            time.sleep(2)

    # Fallback: mark all as unknown
    print(f"  ❌ Failed after 3 attempts, marking batch as unknown")
    return [{"name": n, "is_person": None, "confidence": "low"} for n in names]


def main():
    print(f"Loading {SPEAKER_NAMES_FILE}...")
    df = pd.read_csv(SPEAKER_NAMES_FILE, dtype=str)
    print(f"  {len(df):,} rows loaded.")

    # Add columns if not present
    if "is_person" not in df.columns:
        df["is_person"]            = None
        df["is_person_confidence"] = None

    # Find rows not yet classified
    todo_mask = df["is_person"].isna()
    todo_idx  = df[todo_mask].index.tolist()
    print(f"  {len(todo_idx):,} rows to classify, "
          f"{(~todo_mask).sum():,} already done.")

    if not todo_idx:
        print("All rows already classified!")
        return

    client = OpenAI(base_url=BASE_URL, api_key="lm-studio")

    # Process in batches
    batches     = [todo_idx[i:i+BATCH_SIZE] for i in range(0, len(todo_idx), BATCH_SIZE)]
    total_done  = 0

    for batch_num, batch_idx in enumerate(tqdm(batches, desc="Classifying")):
        names   = df.loc[batch_idx, "name_cleaned"].fillna("").tolist()
        results = classify_batch(client, names)

        for i, (idx, result) in enumerate(zip(batch_idx, results)):
            df.at[idx, "is_person"]            = result.get("is_person")
            df.at[idx, "is_person_confidence"] = result.get("confidence", "high")

        total_done += len(batch_idx)

        # Save progress periodically
        if (batch_num + 1) % SAVE_EVERY == 0:
            df.to_csv(SPEAKER_NAMES_FILE, index=False)
            tqdm.write(f"  💾 Saved progress ({total_done:,} classified so far)")

    # Final save
    df.to_csv(SPEAKER_NAMES_FILE, index=False)

    # Summary
    print(f"\nDone! Classification summary:")
    print(f"  Persons:     {(df['is_person'] == 'True').sum():,}")
    print(f"  Non-persons: {(df['is_person'] == 'False').sum():,}")
    print(f"  Unknown:     {df['is_person'].isna().sum():,}")
    print(f"\nBy dataset type:")
    print(df.groupby("source_dataset_type")["is_person"].value_counts().to_string())


if __name__ == "__main__":
    main()
