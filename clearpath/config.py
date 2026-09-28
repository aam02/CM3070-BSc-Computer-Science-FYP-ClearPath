"""Shared paths and constants for ClearPath."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
SHAP_DIR = ARTIFACTS / "shap"
METRICS_DIR = ARTIFACTS / "metrics"
FIGURES_DIR = ARTIFACTS / "figures"

MODULES = ("BBB", "DDD", "FFF")
PRESENTATIONS = ("2013B", "2013J", "2014B", "2014J")
TRAIN_PRESENTATIONS = ("2013B", "2013J")
TEST_PRESENTATIONS = ("2014B", "2014J")

MIN_ITEMS_PER_STUDENT = 10
MIN_STUDENTS_PER_ITEM = 20
CANDIDATE_K = 100
TOP_N = 10
TEMPORAL_HOLDOUT_FRAC = 0.2
RANDOM_SEED = 42

SPLIT_B_TRAIN_FRAC = 0.70
SPLIT_B_VAL_FRAC = 0.15
SPLIT_A_VAL_FRAC = 0.15

TOP_ACTIVITY_TYPES = (
    "resource",
    "oucontent",
    "forumng",
    "quiz",
    "url",
    "subpage",
    "glossary",
    "oucollaborate",
)

REQUIRED_CSVS = (
    "studentVle.csv",
    "vle.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "courses.csv",
)

RANKER_PARAM_GRID = (
    {"num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 40},
    {"num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 50},
    {"num_leaves": 31, "learning_rate": 0.1, "min_data_in_leaf": 30},
    {"num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 80},
)
RANKER_SEEDS = (0, 1, 2, 3, 4)
