import torch
import numpy as np
import random
from train import Trainer
from config import MODEL_PATH, DATA_PATH_PREFIX, SEED

# Set seed for reproducibility and other boilerplate

if SEED is None:
    SEED = random.randint(0, 10000)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load data ###
    #model parameters are all autocompleted in trainer init based on the data,
    #test data is also set up in the model
    #instantiate a new model with the weights in the model path weights
    #predict_mode=True ensures that we use the given model
predictor = Trainer(model_path=MODEL_PATH, data_path_prefix=DATA_PATH_PREFIX, predict_mode=True)

#### Inference ###
preds, labels, embeddings = predictor.predict()

### Visualise ###
predictor.visualise(embeddings, labels)
