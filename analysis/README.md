# Analysis

Notebooks and shared code for analysing the two analysis-ready datasets. This
tree is **read-only** with respect to the pipeline: it consumes the parquets in
`/home/tom/data/analysis/` and never writes back to the enrichment / target /
party-linkage databases.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
nbstripout --install          # strip notebook outputs on commit -> clean diffs
jupyter lab
```

If the parquets live somewhere other than `/home/tom/data/analysis`, set
`ANALYSIS_DIR` before launching Jupyter.

## Layout

Folders follow the paper's hypotheses:

| Folder | Hypotheses | Design |
|---|---|---|
| `lib/` | — | Shared code: loaders, codebooks, plot style, **`panel.py`** (speaker-year panel). |
| `00_overview/` | — | Descriptives, coverage, sanity checks. |
| `01_trends/` | H1 / H1b | Country-year accusation rate; Poisson trend with country FE + exposure offset. |
| `02_populism/` | H2a / H2b | Speaker-year NB models, made & received. **Modelling template for 03/05.** |
| `03_individual/` | H3a–c | Gender & education, same skeleton as 02. |
| `04_retaliation/` | H3d | Directed dyads: does A→B predict B→A within a window vs. baseline? |
| `05_party/` | H4a–d | Ideology (left_right) & incumbency (in_cabinet), same skeleton as 02. |
| `outputs/` | — | Exported figures & tables (gitignored). |

Most models run on the **speaker-year panel** — build it once (aggregates the
149M-row corpus, takes a few minutes):

```bash
cd analysis && python3 -m lib.panel
```

## Excluded countries

Four countries are dropped from **every** analysis, mirroring the LieLines
validation sampler (`LieLines-Validation/sample_random.py`): **Iceland (IS),
Bosnia & Herzegovina (BA), Greece (GR), Latvia (LV)** — see
`codebooks.EXCLUDED_COUNTRIES`. All three loaders apply this filter by default,
so you never have to remember it. To inspect the raw, unfiltered data pass
`exclude_countries=False` (e.g. `data.load_accusations(exclude_countries=False)`).

## Loading data

Every notebook (in any subfolder) starts with:

```python
import sys; sys.path.append("..")        # so `lib` is importable
from lib import data, codebooks, viz
viz.apply_style()

acc = data.load_accusations()            # 532k rows, pandas — fine to load whole
```

For the **149M-row full corpus**, never `read_parquet` it whole. Use lazy polars:

```python
import polars as pl
lf = data.scan_corpus()
by_country = (lf.group_by("country")
                .agg(pl.col("lie_score").cast(pl.Float64).mean().alias("lie_rate"))
                .collect())                  # only the small result materialises
```

…or DuckDB SQL:

```python
con = data.duck()
con.execute("SELECT country, AVG(lie_score) FROM corpus GROUP BY country").df()
```

## The two datasets

### `accusations_dataset.parquet` — 532,268 rows, one per accusation

- **Accusation core**: `id, source_dataset, country, date, sentence, context, lie_score`,
  `is_interjection, interjector_raw`.
- **Accuser** (`accuser_*`): the actual accuser — the resolved interjector for
  interjections, otherwise the recorded speaker. `accuser_match` ∈
  {`resolved`, `no_party`, `no_speaker`}. When `resolved`: individual-level
  (`gender, birth_year, age, birth_place, highest_isced, career_sectors`), party
  (`partyfacts_id, party_name`), and party-level (`vote_share_*, in_cabinet,
  is_pm_party, years_since_government, left_right, populism, anti_elitism,
  people_centrism`).
- **Target** (`target_*`): `target_type` (Phase-1 scheme, see `codebooks.TARGET_TYPE`),
  `target_text`, `resolve_status`, `resolved_speaker_id`. When the target resolved
  to a known speaker (`target_match == "resolved"`, 82,684 rows) the same full
  individual + party-level var set is present under `target_*`.

### `full_corpus_dataset.parquet` — 149,407,898 rows, one per sentence

Every sentence in the scored corpus with `lie_score` retained and the speaking
MP's full variable set attached under `speaker_*` (same schema as `accuser_*`
above). `speaker_match` tells you whether the speaker/party resolved.

## Value labels

`lib/codebooks.py` holds the label maps: `ISCED` (5–8), `SECTOR` (1–8, national MP
mandate excluded), `TARGET_TYPE`, `GENDER`. Use `codebooks.sectors_to_labels(...)`
to expand the comma-joined `career_sectors` field.
