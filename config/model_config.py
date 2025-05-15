"""
Configuration settings for the wine quality prediction model.
"""

# Data settings
DATA_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
TEST_SIZE = 0.25
VALID_SIZE = 0.2
RANDOM_STATE = 42

# Model architecture
HIDDEN_LAYERS = [64, 32]
DROPOUT_RATES = [0.2, 0.1]

# Training settings
BATCH_SIZE = 64
MAX_EPOCHS = 3
MAX_EVALS = 8  # Number of hyperparameter optimization trials

# Hyperparameter search space
LEARNING_RATE_MIN = 1e-5
LEARNING_RATE_MAX = 1e-1
MOMENTUM_MIN = 0.0
MOMENTUM_MAX = 1.0

# MLflow settings
EXPERIMENT_NAME = "/wine-quality"
MODEL_NAME = "wine-quality-predictor" 