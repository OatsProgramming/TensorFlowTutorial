import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from os import system

system('clear')

'''
Although we've preprocessed data before, there are still some things to improve this first step
Let's first list out what we do know abt preprocessing data
    1) Turn all data into numbers (NN cant handle strings)
    2) Make sure all of your tensors are the right shape
But there's another thing that we're missing
    3) Scale features (normalize or standardize, NN tend to prefer normalization)

What is Normalization?
    A technique often applied as part of data preparation for ML
    The goal of normalization is to change the values of numeric columns in the dataset to a
    common scale, w/o distorting differences in the ranges of values

    Ex:
        We have one dataset (age) that ranges from 1 to 60
        Another (children) that ranges from 0 to 5
        Another (bmi) that ranges from 15 to 40
        And so on...
        But what if we want a more general overview of that by having them all set to a range of 0 to 1?
        Thats what normalization does

Feature Scaling:
    Scale (AKA normalization):
        What it does:           Converts all values to btwn 0 and 1 whilst preserving the original distribution
        Scikit-Learn Function:  MinMaxScaler
        When to use:            Use as default scaler w/ NN
    
    Standardization:
        What it does:           Removes the mean and divides each value by the standard deviation
        Scikit-Learn Function:  StandardScaler
        When to use:            Transform a feature to have close to normal distribution
                                (CAUTION: this reduces the effects of outliers )

If you're not sure on which to use, you could try to use both and see which performs better                              
'''

# Set global seed
tf.random.set_seed(42)

# Get our datasets
insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')

# To prepare our data, we can borrow from Scikit-Learn
# from sklearn.compose import make_column_transformer
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Now create a column transformer
ct = make_column_transformer(
    (MinMaxScaler(), ['age', 'bmi', 'children']), # Turn all values in these columns btwn 0 and 1
    (OneHotEncoder(handle_unknown = 'ignore'), ['sex', 'smoker', 'region']) # 'ignore' if theres any columns that onehot doesnt know, ignore it
)

# Create X and y
X = insurance.drop('charges', axis = 1)
y = insurance['charges']

# Train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fit the column transformer to our training data
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScaler) and OneHotEncoder
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# What does our data look like now?
print(X_train.loc[0])
print("X TRAIN")
print(X_train_normal[0])
print("X TRAIN NORMAL")

# How has our shape changed?
print(X_train.shape)
print('X TRAIN SHAPE')
print(X_train_normal.shape)
print('X TRAIN NORMAL SHAPE')