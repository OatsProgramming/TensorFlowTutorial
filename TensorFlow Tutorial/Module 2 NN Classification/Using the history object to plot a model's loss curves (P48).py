import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from os import system

system('clear')

# Evaluating and Improving our Classification Model

# Create a plot decision boundary for visualization
def plot_decision_boundary(model, x, y):
    '''
    Plots the decision boundary created by a model prediction on X
    '''

    # Define the axis boundaries of the plot
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()

    # Create a meshgrid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Create X value (Going to make predictions on these)
    x_in = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions
    y_pred = model.predict(x_in)

    # Check if its a multiclass classification
    if len(y_pred[0]) > 1:
        print('This is a multiclass classification')
        y_pred = np.argmax(y_pred).reshape(xx.shape)
    else:
        print('This is a binary classification')
        y_pred = np.round(y_pred).reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap = plt.cm.RdYlBu, alpha = 0.7)
    plt.scatter(x[:, 0], x[:, 1], c = y, s = 40, cmap = plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    

# Set seed
tf.random.set_seed(42)

# Create circles
X, y = make_circles(
    n_samples=1000,
    noise=0.03,
    random_state=42
)

# Create compile and fit latest model

model_7 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(4, activation = 'relu'),
    tf.keras.layers.Dense(4, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model_7.compile(
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    metrics = ['accuracy']
)

history = model_7.fit(X, y, epochs = 100, verbose = 0)

#plot_decision_boundary(model_7, X, y)

# Lets get our training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Now lets try to implement this into our model and evaluate it

history_7 = model_7.fit(X_train, y_train, epochs = 25, verbose = 0)

print('\nEVALUATION\n')
model_7.evaluate(X_test, y_test)

# Plot the decision boundaries for the training and test sets to visualize
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)    # 1 row 2 columns the first value is the Training plot
plt.title('Train')
plot_decision_boundary(model_7, X_train, y_train)
plt.subplot(1, 2, 2)    # 1 row 2 columns the second plot is the Testing plot
plt.title("Test")
plot_decision_boundary(model_7, X_test, y_test)
#plt.show()

'------------------------------------------------------------'
'------------------------------------------------------------'
'------------------------------------------------------------'

# Plot the loss (aka training) curves to visualize

# What does fit() actually do?
#   Returns history objects:
#       History.history attribute is a record of training loss values and metric values at successive
#       as well as validation loss values and validation metrics values (if applicable)

print('\nHISTORY OF LOSS VALUES AND METRICS VALUES:\n')
print(history_7.history)

# Since we can't exactly understand the history object rn
# Lets turn it into a Dataframe
print('\nHISTORY DATAFRAME:\n')
dataframe = pd.DataFrame(history_7.history)
print(dataframe)

# Plot the loss curves
dataframe.plot()
plt.title('Model_7 Loss Curves')
plt.show()

# The plot should be intuitive