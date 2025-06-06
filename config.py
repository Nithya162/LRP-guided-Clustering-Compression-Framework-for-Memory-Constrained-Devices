import torch
import os

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seeds
SEED = 42

# Data configuration
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 1000

# Training configuration
EPOCHS = 5
LEARNING_RATE = 0.001

# LRP configuration
LRP_EPSILON = 1e-7
LRP_GAMMA = 0.05
NUM_LRP_BATCHES = 10

# Clustering configuration
PCA_VARIANCE_THRESHOLD = 0.95
K_MIN = 2
K_MAX_RATIO = 0.25  # K_max = min(50, N_neurons * K_MAX_RATIO)

# K-rep weights
W_ALPHA = 1.0
W_BETA = 1.0
W_DELTA = 1.0
W_ETA = 1.0

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PHASE1_DIR = os.path.join(RESULTS_DIR, "phase1_lrp")
PHASE2_DIR = os.path.join(RESULTS_DIR, "phase2_clustering")
PHASE3_DIR = os.path.join(RESULTS_DIR, "phase3_quantization")
PHASE4_DIR = os.path.join(RESULTS_DIR, "phase4_compression")
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, "saved_models")

# Create directories
for dir_path in [RESULTS_DIR, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE4_DIR, MODEL_SAVE_DIR]:
    os.makedirs(dir_path, exist_ok=True)