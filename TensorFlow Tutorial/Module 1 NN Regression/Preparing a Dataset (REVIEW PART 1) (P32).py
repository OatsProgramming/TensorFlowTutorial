import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from os import system

system('clear')

# Let's recreate what we've done so far but with a larger example
# To do so, lets look at medical cost datasets from kaggle.com

# We're going to get all the data to predict the mdeical costs billed by health insurance
# 'Can you accurately predict insurance costs?'

# Read in the insurance dataset
insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
print(insurance, 'DATAFRAME')

# Reminder: what is regression?
#   Try and figure out the relationship btwn a dependent variable (label) and one or more independent variable (features)

# For this example, everything but charges (label) will be our features

# In the example, we have some data that are numerical datatypes and others non-numerical datatypes

# Before going forward, we have to encode the non-numerical datatypes into numerical datatypes in order for our Algorithm to read it

# In order to do so, let's implement a method of encoding we've learned: one-hot encoding

# So how do we one-hot encode a pandas Dataframe?
# pandas.get_dummies()

insurance_one_hot = pd.get_dummies(insurance)
print(insurance_one_hot, 'ONE HOT ENCODED')