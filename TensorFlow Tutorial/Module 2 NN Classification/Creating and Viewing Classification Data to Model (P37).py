import tensorflow as tf
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
from os import system

system('clear')

# A classification is where you try to classify something as one thing or another
# Few types of classification problems:
#   Binary classification
#   Multiclass classification
#   Multilabel classification


# Creating data to view and fit
# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(
    n_samples,
    noise = 0.03,
    random_state = 42
)

# Check out the features
print('\nFEATURES\n', X)

# Check the labels 
print('\nLABELS\n', y)

# Our data is a little hard to understand rn; lets visualize it
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
print('\nDATAFRAME\n', circles)

# Visualize with a plot
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.RdYlBu)
plt.show()

# For more depth, check out tensorflow playground
# https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.18103&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false