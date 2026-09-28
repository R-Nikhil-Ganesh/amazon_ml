"""
Business Entity Resolution System (Amazon ML Challenge 2026).

A high-performance, 3-stage cascaded hybrid entity resolution pipeline:
- Stage 1: Disk-backed SQLite & In-Memory Multi-Pass Blocking (< 400 MB RAM)
- Stage 2: LightGBM Tabular Match Screener (Fast, High-Precision filtering)
- Stage 3: Deep Residual Neural Network (TensorFlow/Keras 3) Specialist for Ambiguous Pairs
"""

from .config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    MODELS_DIR,
    OUTPUT_DIR,
)
from .normalize import (
    normalize_business_name,
    normalize_address,
    extract_numbers,
)
from .blocking import BlockingEngine, read_entity_csv
from .sqlite_blocking import DiskBlockingEngine
from .features import extract_pair_features, FEATURE_NAMES
from .evaluate import evaluate_predictions, load_mapping_file

__version__ = "1.0.0"

__all__ = [
    "BlockingEngine",
    "DiskBlockingEngine",
    "read_entity_csv",
    "extract_pair_features",
    "FEATURE_NAMES",
    "normalize_business_name",
    "normalize_address",
    "extract_numbers",
    "evaluate_predictions",
    "load_mapping_file",
]

