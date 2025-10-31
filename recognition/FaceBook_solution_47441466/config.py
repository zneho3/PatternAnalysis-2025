#File to change global configuration variables and hyperparameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.10
TEST_RATIO = 0.10
LEARNING_RATE = 0.01

SEED = 42  # For reproducibility, set to None for random seed
NUM_EPOCHS = 200
LOSS_BUFFER = 10  # Number of epochs to consider for early stopping

#Set these paths according location of where you have stored the data and model, e.g. in Google Drive if using Colabs
#Saved model will be saved to MODEL_PATH after training and then loaded from there for prediction
MODEL_PATH = '/content/drive/MyDrive/COMP3710/A3/PatternAnalysis-2025/recognition/Facebook_solution_47441466/Saved_models/model.pth'
DATA_PATH_PREFIX = '/content/drive/MyDrive/COMP3710/A3/PatternAnalysis-2025/recognition/Facebook_solution_47441466/facebook' 

