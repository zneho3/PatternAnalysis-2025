# Solution for Problem 4 - Facebook Large Page-Page Network Dataset

## Dataset Information

This repository contains the solution for Problem 4 of the assignment, which involves analyzing the Facebook Large Page-Page Network Dataset. The dataset can be found at the following link: [Facebook Large Page-Page Network Dataset](https://snap.stanford.edu/data/facebook-large-page-page-network.html).

The original dataset to be used as outlined in the spec sheet was in a .npz file, which had simplified features reduced to 128 dimensions. However, this file was unavailable. Instead, the raw dataset in csv format was used, which did not have an impact on performance or training times.The dataset consists of three csv files: `musae_facebook_edges.csv`, `musae_facebook_target.csv`, and `musae_facebook_features.csv`. The edges file contains the connections between nodes, the target file contains the labels for each node, and the features file contains the feature vectors for each node.  Minimal preprocessing was performed, outside of converting the csv files into a graph structure, outside of splitting the data into training, validation, and test sets using masking.


## Model Description

The model used for this problem is a Graph Neural Network (GNN) implemented using the PyTorch Geometric library. The architecture consists of two graph convolutional layers using a ReLu layer in between.Originally the plan was to use a more complicated architecture with more layers but this simple architecture worked quite well upon initial creation and testing so it was kept.

## Model Training

Data was split into training, validation, and test sets using ratios of 80%, 10%, and 10% respectively. The model was trained for 200 epochs using the Adam optimizer with a learning rate of 0.01.

## Usage Instructions

Download the dataset from the provided link and place it in the same directory as the code files. Ensure that the file names are not changed from the source, and that the files are formatted as such:

```
facebook
├── musae_facebook_edges.csv
├──musae_facebook_target.csv
└──musae_facebook_features.csv
```

 Run the `train.py` script to train the model. After training, use the `predict.py` script to evaluate the model on the test set and visualize the results. Alternatively, you can run the `train.ipynb` and then `predict.ipynb` Jupyter notebooks to use Google Colab GPU resources.

Set config options in the `config.py` file as needed for file paths, and hyperparameters.

## Requirements

If using the python scripts, ensure that you have the required libraries installed. You can install them using pip:

```bash
pip install -r requirements.txt`
```

The required libraries include:

```bash
matplotlib
networkx
numpy
pandas
scikit_learn
torch
torch_geometric
```

These are automatically installed when using the Jupyter notebooks in Google Colab.

## Findings

Different combinations of data splits and learning rates were experimented with, however the initial test of 80:10:10 worked the best, along with a learning rate of 0.01. This configuration yielded the highest accuracy on the test set of 94.88 percent.


### Configurations and Results

The table below summarizes the different configurations tested and their corresponding accuracies on the test set.
| Train:Val:Test Ratio | Learning Rate | Test Set Accuracy (%) |
|----------------------|---------------|-----------------------|
| 80:10:10             | 0.01          | 94.88


## Conclusion

## Project note

Lots of OOP was used in this project to keep the code modular and reusable. Maybe more than necessary but I still wanted to do it like this. Especially since I was working with Colab.

## References

