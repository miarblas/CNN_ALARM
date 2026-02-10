# ==========================
# PREPROCESSING
# ==========================
WINDOW_SIZE = 20  # segundos de ventana temporal
TARGET_FS = 125  # frecuencia de muestreo objetivo (Hz)
MAX_CHANNELS = 4  # número máximo de canales

# ==========================
# MODEL ARCHITECTURE
# ==========================
MODEL_BASE = 32  # filtros base de la CNN
MODEL_DROPOUT = 0.3  # dropout rate

# ==========================
# TRAINING
# ==========================
BATCH_SIZE = 32
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 3e-4
EPOCHS = 100  # Max epochs
PATIENCE = 25 # Early stopping patience
POS_WEIGHT = 2.8  # For class imbalance

# ==========================
# DATA SPLIT - ONLY TRAIN/VAL FROM PROVIDED TRAIN SET
# ==========================
VAL_FRACTION = 0.2  # Only for validation during training
RANDOM_SEED = 42  # semilla para reproducibilidad

# ==========================
# THRESHOLD TUNING
# ==========================
THRESHOLD_TUNING_METRIC = 'f1_weighted'  # Options: 'f1', 'f1_weighted', 'youden', 'balanced_acc'
DEFAULT_THRESHOLD = 0.35  # Default before tuning

# ==========================
# PATHS
# ==========================
ARTIFACTS_DIR = 'artifacts'
DATA_DIR = 'data'
MODEL_WEIGHTS = 'model.pth'
BEST_MODEL_WEIGHTS = 'best_model.pth'
CONFIG_FILE = 'config.json'