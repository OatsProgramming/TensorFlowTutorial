import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from os import system

system('clear')

'''
Remember, it's important to VISUALIZE EVERYTHING:
    Data
    Model
    Training Data
    Predicition Data
'''

# Set seed
tf.random.set_seed(42)

# Create circles
X, y = make_circles(
    n_samples=1000,
    noise=0.03,
    random_state=42
)

# Create, compile, fit, evaluate

model_2 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1),
])

model_2.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(lr = 0.1),
    metrics = ['accuracy']
)

model_2.fit(X, y, epochs = 100, verbose = 0)

print('\nEVALUATION\n')
model_2.evaluate(X, y)

'------------------------------------------------------'

# Lets check out the predicitions that our model is making
# print(model_2.predict(X))

# After trying this out, you will see that it's a bit difficult to understand with the massive amnts of data
# So we'll visualize instead

# To visualize our model's prediction, lets create a function: plot_decision_boundary().config/
# This function will:
#   Take in a trained model, features(X), labels(y)
#   Create a meshgrid of the different X values
#   Make predictions across the meshgrid
#   Plot the predictions as well as a line btwn zones (where each unique class falls)

def plot_decision_boundary(model, X, y):
    '''
    Plots the decision boundary created by a model predicition on X
    '''
    # Define the axis boundaries of the plot
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1 # '0.1' will give us some margin
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # Create the meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                        np.linspace(y_min, y_max, 100))

    # Create X value (we're going to make predictions on these)
    x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together

    # Make predictions
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) > 1:
        print('Doing multiclass classification')
        # We have to reshape our prediction if multiclass
        y_pred = np.argmax(y_pred, axis = 1).reshape(xx.shape)
    else:
        print('Doing binary classification')
        y_pred = np.round(y_pred).reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap = plt.cm.RdYlBu, alpha = 0.7)
    plt.scatter(X[:, 0], X[:, 1], c = y, s = 40, cmap = plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# Check out the predictions our model is making
plot_decision_boundary(model_2, X, y)

# To learn more abt plot decision boundary check 
#   CS231n Convolutional Neural Networks for Visual Recognition
#   Made with ML

