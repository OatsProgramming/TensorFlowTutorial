import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from os import system

system('clear')

# Set up

# Set seed for reproducibility
tf.random.set_seed(42)

# Get our data
insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')

# Column transform and onehot encode it
ct = make_column_transformer(
    (MinMaxScaler(), ['age', 'bmi', 'children']),
    (OneHotEncoder(handle_unknown = 'ignore'), ['sex', 'smoker', 'region'])
)

# Create feature and label
X = insurance.drop('charges', axis=1)
y = insurance['charges']

# Create training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fit the column transformer into our data
ct.fit(X_train)

# Transform Training and Testing data with normalization
X_train_normalize = ct.transform(X_train)
X_test_normalize = ct.transform(X_test)



'----------------------------------------------------------------------------'
# LESSON STARTS BELOW

# Build a NN model to fit our normalized data

insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr=0.01),
    metrics = ['mae']
)

insurance_model.fit(X_train_normalize, y_train, epochs = 100)

# Evaluate our insurance model trained on normalized data
print("\nEVALUATION")
insurance_model.evaluate(X_test_normalize, y_test)

