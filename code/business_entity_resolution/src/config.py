#!/usr/bin/env python3
"""
Central configuration for Business Entity Resolution pipeline.
Configured to use CSV datasets while generating compliant TSV competition submissions.
"""

import os

def delimiter_for(path: str) -> str:
    """Field delimiter by extension: the competition's .tsv files are tab-separated,
    the derived .csv files (split/hard-negative caches) are comma-separated."""
    return "\t" if str(path).lower().endswith(".tsv") else ","


# Base Directories
# Defaults to the submission root (three levels above src/); set ER_PROJECT_ROOT to run
# from elsewhere. Expects <root>/student_resource/dataset/{train,test}/*.tsv.
PROJECT_ROOT = os.path.abspath(os.environ.get("ER_PROJECT_ROOT") or os.path.join(os.path.dirname(__file__), "../../.."))
DATA_DIR_TRAIN = os.path.join(PROJECT_ROOT, "student_resource/dataset/train")
DATA_DIR_TEST = os.path.join(PROJECT_ROOT, "student_resource/dataset/test")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data_split")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset Paths (CSV files)
TRAIN_SOURCE1_PATH = os.path.join(DATA_DIR_TRAIN, "train_source1.tsv")
TRAIN_SOURCE2_PATH = os.path.join(DATA_DIR_TRAIN, "train_source2.tsv")
TRAIN_SOURCE3_PATH = os.path.join(DATA_DIR_TRAIN, "train_source3.tsv")
TRAIN_GROUND_TRUTH_PATH = os.path.join(DATA_DIR_TRAIN, "train_ground_truth.tsv")

TEST_SOURCE1_PATH = os.path.join(DATA_DIR_TEST, "test_source1.tsv")
TEST_SOURCE2_PATH = os.path.join(DATA_DIR_TEST, "test_source2.tsv")
TEST_SOURCE3_PATH = os.path.join(DATA_DIR_TEST, "test_source3.tsv")

# DuckDB blocking index built over the FULL training S2/S3 pool, used by
# DuckDBBlockingEngine - shared by
# validate_pipeline.py and train_screener.py/train_tf_specialist.py so both
# ends of training (candidate retrieval for model training, and validating
# the trained model) see the same index.
TRAIN_CANDIDATE_INDEX_DUCKDB_PATH = os.path.join(SPLIT_DIR, "train_candidates_index.duckdb")

# Submission Paths (TSV format required by challenge)
SUBMISSION_MATCHING_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
SUBMISSION_CANDIDATE_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Validation Split Paths
VAL_GROUND_TRUTH_PATH = os.path.join(SPLIT_DIR, "val_ground_truth.tsv")
MINI_VAL_GROUND_TRUTH_PATH = os.path.join(SPLIT_DIR, "mini_val_ground_truth.tsv")
VAL_S1_IDS_PATH = os.path.join(SPLIT_DIR, "val_s1_ids.txt")
MINI_VAL_S1_IDS_PATH = os.path.join(SPLIT_DIR, "mini_val_s1_ids.txt")

# Blocking Hyperparameters
MAX_CANDIDATES_PER_S1 = 35
MAX_TOKEN_POSTINGS = 150  # Cap on posting list length to ignore super-frequent tokens

# retrieve_candidates() pulls the full key-matched pool, then reranks the top
# RERANK_POOL_SIZE of it by text similarity (not just summed key weight)
# before truncating to MAX_CANDIDATES_PER_S1 - see build_blocking_keys() /
# retrieve_candidates() in sqlite_blocking.py. 300 covers the p99 pool size
# (249) measured on the real train index while keeping the rapidfuzz cost
# per S1 entity small.
RERANK_POOL_SIZE = 300

# Common Stopwords / Frequency Filters (Multilingual)
STOPWORDS = {
    # English
    "the", "and", "of", "in", "for", "on", "at", "to", "a", "an", "by", "with",
    "co", "inc", "ltd", "corp", "llc", "company", "corporation", "limited",
    "services", "solutions", "enterprises", "group", "associates",
    # India
    "pvt", "shree", "sri", "om", "india", "private", "trading",
    # French
    "de", "la", "le", "les", "et", "du", "des", "en", "au", "aux", "pour",
    "sarl", "sas", "sa", "france", "societe",
}

# LightGBM Classifier Parameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "n_estimators": 400,
    "n_jobs": 8,
    "random_state": 42,
    "verbose": -1,
}

# Cascaded Routing Probability Thresholds
THRESHOLD_AUTO_ACCEPT = 0.88   # Highly confident matches
THRESHOLD_AUTO_REJECT = 0.15   # Highly confident non-matches
# Ambiguous zone routed to the Stage 3 TensorFlow specialist - see pipeline.py.
