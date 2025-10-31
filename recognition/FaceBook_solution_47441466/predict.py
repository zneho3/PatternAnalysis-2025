# from train import Trainer
# from dataset import FacebookData
# from modules import GCN

# data_path = 'data/facebook_combined.txt'
# model_path = 'models/facebook_gcn.pth'
# output_path = 'results/facebook_inference.txt'
print("Starting imports")
from modules import GCN
import torch
from dataset import FacebookData
from train import Trainer
from config import SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, NUM_EPOCHS
import numpy as np
import random
print("Imports done")
# Set seed for reproducibility and other boilerplate
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load data
print("Setting up model")
model_path = "G:/My Drive/COMP3710/A3/PatternAnalysis-2025/recognition/FaceBook_solution_47441466/Saved_models/model.pth"
data_path = "G:/My Drive/COMP3710/A3/facebook"

#model parameters are all autocompleted in trainer init based on the data, 
#test data is also set up in the model
#instantiate a new model with the weights in the model path weightss

predictor = Trainer(model_path=model_path, data_path_prefix=data_path)
#model setup
#inference
# predictor.print_data_shapes()
print("Starting prediction")
preds, labels = predictor.predict()
# visualise

predictor.visualise(preds, labels)