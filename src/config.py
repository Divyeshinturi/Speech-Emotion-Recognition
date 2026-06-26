
from pathlib import Path

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

RAVDESS_PATH = DATA_DIR / "RAVDESS"
CREMAD_PATH = DATA_DIR / "CREMA-D" / "AudioWAV"

# ==========================
# Audio Processing
# ==========================

SAMPLE_RATE = 22050

# ==========================
# Feature Extraction
# ==========================

N_MFCC = 40
N_CHROMA = 12
N_MELS = 128

# ==========================
# Data Split
# ==========================

TEST_SIZE = 0.20
RANDOM_STATE = 42

# ==========================
# Training
# ==========================

EPOCHS = 50
BATCH_SIZE = 32

# ==========================
# Augmentation
# ==========================

USE_AUGMENTATION = True
ADD_NOISE = True
PITCH_SHIFT = True
TIME_STRETCH = True

# ==========================
# Supported Datasets
# ==========================

SUPPORTED_DATASETS = [
    "RAVDESS",
    "CREMA-D"
]