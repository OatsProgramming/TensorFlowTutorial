import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os import system

system('clear')

# Set global seed
tf.random.set_seed(42)

# Get our datasets
insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')

# One hot encode any non-numerical data
insurance_one_hot = pd.get_dummies(insurance)

# Get features and labels
X = insurance_one_hot.drop('charges', axis = 1) # axis set to one for column data
y = insurance_one_hot['charges']

# Get training and testing dataset
# Remember: Xtrain, Xtest, ytrain, ytest IN THAT ORDER
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(len(X_test))
print(len(y_train))

# Now, with everything set up, referring from the last review, how can we improve our model to lower loss?

insurance_model_2 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model_2.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    metrics = ['mae']
)

# Check below codes to see why we're using a history variable
history = insurance_model_2.fit(X_train, y_train, epochs = 100, verbose = 0)

# Evaluate the model to see if there's an improvement
insurance_model_2.evaluate(X_test, y_test)

# Plot history (AKA loss curve or training curve)
# This will illustrate the effectiveness of the algorithm
pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

# QUESTION: How long should you train for?
#   It depends. Many ppl have asked this question before and thus a 'solution' was given
#   A TensorFlow component, EarlyStoppingCallback, you can add to your model to stop training
#   once it stops improving at a certain metric