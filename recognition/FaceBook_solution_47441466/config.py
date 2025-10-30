#File to change global configuration variables and hyperparameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO

SEED = 42  # For reproducibility
NUM_EPOCHS = 50
LEARNING_RATE = 0.01