"""
config.py — paths + design params for party-ideology linkage.
"""

DATA_DIR      = "/home/tom/data"

PARLGOV_DIR   = f"{DATA_DIR}/parlgov"
ELECTION_CSV  = f"{PARLGOV_DIR}/view_election.csv"
CABINET_CSV   = f"{PARLGOV_DIR}/view_cabinet.csv"
PARTY_CSV     = f"{PARLGOV_DIR}/view_party.csv"

VPARTY_CSV    = f"{DATA_DIR}/vparty/CPD_V-Party_CSV_v2/V-Dem-CPD-Party-V2.csv"
CROSSWALK_CSV = f"{DATA_DIR}/partyfacts/partyfacts-external-parties.csv"

SPEAKER_NAMES = f"{DATA_DIR}/speaker_names.csv"
ENRICH_DB     = f"{DATA_DIR}/speaker_enrichment/speaker_enrichment.db"
ACC_DB        = f"{DATA_DIR}/target_detection/accusations.db"

# ParlGov election_type value for national parliamentary elections
NATIONAL_ELECTION = "parliament"

# Lielines-scored sentence corpus (source for the full-corpus dataset)
PREDICTED_CSV = f"{DATA_DIR}/sentence_corpus_predicted.csv"

# Analysis-ready outputs
ANALYSIS_DIR        = f"{DATA_DIR}/analysis"
ACCUSATION_PARQUET  = f"{ANALYSIS_DIR}/accusations_dataset.parquet"
FULL_CORPUS_PARQUET = f"{ANALYSIS_DIR}/full_corpus_dataset.parquet"
