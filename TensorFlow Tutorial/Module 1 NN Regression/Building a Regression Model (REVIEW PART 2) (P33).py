import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os import system

system('clear')

# Read the data
insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')

# One-hot encode any non-numerical data for Algorithm
insurance_one_hot = pd.get_dummies(insurance)
print(insurance_one_hot)

# For this review, we're going to:
#   Create X and y values (features and labels, respectively)
#   Create training and test sets
#   Build a NN

# Create features and labels
#   X_age = insurance_one_hot['age']
#   X_female = insurance_one_hot['sex_female']
#   X_male = insurance_one_hot['sex_male']
#   X_bmi = insurance_one_hot['bmi']
#   X_children = insurance_one_hot['children']
#   X_smoker_no = insurance_one_hot['smoker_no']
#   X_smoker_yes = insurance_one_hot['smoker_yes']
#   X_northeast = insurance_one_hot['region_northeast']
#   X_northwest = insurance_one_hot['region_northwest']
#   X_southeast = insurance_one_hot['region_southeast']
#   X_southwest = insurance_one_hot['region_southwest']
#   y_charges = insurance_one_hot['charges']
# Instead of doing all that, there's a much easier way to do this
# Use .drop() to add everything but the undesired element

X = insurance_one_hot.drop('charges', axis=1)
y = insurance_one_hot['charges']

# View X and y to see if it works
print(X.head())
print('X ELEMENTS')
print(y.head())
print('Y ELEMENTS')

# Training and testing datasets
#   X_age_train = X_age[:int(1338*0.8)]
#   X_female_train = X_female[:int(1338*0.8)]
#   X_male_train = X_male[:int(1338*0.8)]
#   X_bmi_train = X_bmi[:int(1338*0.8)]
#   X_children_train = X_children[:int(1338*0.8)]
#   X_smoker_no_train = X_smoker_no[:int(1338*0.8)]
#   X_smoker_yes_train = X_smoker_yes[:int(1338*0.8)]
#   X_northeast_train = X_northeast[:int(1338*0.8)]
#   X_northwest_train = X_northwest[:int(1338*0.8)]
#   X_southeast_train = X_southeast[:int(1338*0.8)]
#   X_southwest_train = X_southwest[:int(1338*0.8)]
#   
#   y_charges_train = y_charges[:int(1338*0.8)]
#   
#   X_age_test = X_age[int(1338*0.8):]
#   X_female_test = X_female[int(1338*0.8):]
#   X_male_test = X_male[int(1338*0.8):]
#   X_bmi_test = X_bmi[int(1338*0.8):]
#   X_children_test = X_children[int(1338*0.8):]
#   X_smoker_no_test = X_smoker_no[int(1338*0.8):]
#   X_smoker_yes_test = X_smoker_yes[int(1338*0.8):]
#   X_northeast_test = X_northeast[int(1338*0.8):]
#   X_northwest_test = X_northwest[int(1338*0.8):]
#   X_southeast_test = X_southeast[int(1338*0.8):]
#   X_southwest_test = X_southwest[int(1338*0.8):]
#   
#   y_charges_test = y_charges[int(1338*0.8):]

# A much better way to do this is:
#   sklearn.model_selection.train_test_split
#       - Split arrays or matricies into random train and test subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Visualize to make sure it works 
print(len(X))
print('NORMAL X')
print(len(X_train))
print('X TRAINING')
print(len(X_test))
print('X TESTING')

# Set seed
tf.random.set_seed(42)

# Create, compile and fit model
insurance_model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    metrics = ['mae']
)

insurance_model.fit(X_train, y_train, epochs = 100)

# Check the results of the insurance model on the test data
insurance_model.evaluate(X_test, y_test)
print('EVALUATION')