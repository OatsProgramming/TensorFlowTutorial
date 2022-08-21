import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from os import system

system('clear')

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(
    n_samples,
    noise = 0.03,
    random_state = 42
)

# Visualize features
print('\nFEATURES\n', X)

# Visualize labels
print('\nLABELS\n', y)

# Visualize with a dataframe
circle = pd.DataFrame({"X0:": X[:, 0], "X1": X[:, 1], "LABELS:": y})
print('\nDATAFRAME\n', circle)

# Visualize with plot
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.RdYlBu)
plt.show()

'-----------------------------------------------------------------------------------'

# Input and output shapes

# Check the shapes of our features and labels
print('\nXSHAPE\n', X.shape) 
print('\nYSHAPE\n', y.shape)

# How many samples we're working w/
print('\nAMNT X\n', len(X))
print('\nAMNT Y\n', len(y))

# View the first example of features and labels
print('\nFIRST FEATURE\n', X[0])
print('\nFIRST LABEL\n', y[0])

