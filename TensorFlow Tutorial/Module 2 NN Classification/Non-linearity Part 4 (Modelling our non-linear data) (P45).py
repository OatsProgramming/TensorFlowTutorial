import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from os import system

system('clear')

# Setting up the plot decision boundary for visualization

def plot_decision_boundary(model, feature, label):
    '''
    Plots the decision boundary created by a model prediction on X
    '''

    # Define the axis boundaries of the plot
    # Note to self: feature[:, 0] == feature[copy row, keep column 0]
    x_min, x_max = feature[:, 0].min(), feature[:, 0].max()
    y_min, y_max = feature[:, 1].min(), feature[:, 1].max()

    # Create a meshgrid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    ) 

    # Create X value (Going to make predictions on these)
    x_in = np.c_[xx.ravel(), yy.ravel()]
    # np.c_[np.array([1,2,3]), np.array([4,5,6])]
    # Output:
    #           array([[1, 4],
    #                  [2, 5],
    #                  [3, 6]])

    # Make predictions
    y_pred = model.predict(x_in)

    # Check if its a multiclass classification
    if len(y_pred[0]) > 1:
        print('This is a multiclass classification')
        y_pred = np.argmax(y_pred, axis = 1).reshape(xx.shape)
    else:
        print('This is a binary classification')
        y_pred = np.round(y_pred).reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap = plt.cm.RdYlBu, alpha = 0.7)
    plt.scatter(feature[:,0], feature[:, 1], c = label, s = 40, cmap = plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

    
# Set seed
tf.random.set_seed(42)

# Create circles
X, y = make_circles(
    n_samples = 1000,
    noise = 0.03,
    random_state = 42
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

#plot_decision_boundary(model_2, X, y)

# Lets see if our model can be used for a regression problem...

# Create some regression data
X_regression = tf.range(0, 1000, 5)
y_regression = tf.range(100, 1100, 5) # Relationship: y = X + 100

# Visualize the data
print('\nREGRESSION X\n', X_regression)
print('\nREGRESSION Y\n', y_regression)

# Split our regression data into training and test sets
X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]

y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]

# Fit our model to the regression data

# The following won't work bc our model_2 is a binary classification model
# And we're trying to fit a regression; we're supposed to use loss = mae or mse
# model_2.fit(tf.expand_dims(X_reg_train, -1), y_reg_train, epochs = 100)

model_3 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1),
])

# Now, instead of binary, we use a regression specific model
model_3.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr = 0.1),
    metrics = ['mae']
)

model_3.fit(tf.expand_dims(X_reg_train, -1), y_reg_train, epochs = 100, verbose = 0)

print('\nMODEL 3 EVALUATION\n')
model_3.evaluate(X_reg_test, y_reg_test)

# Make a predictions with our training model
y_reg_pred = model_3.predict(X_reg_test)

# Plot the model's prediction against our regression data
#plt.figure(figsize = (10, 7))
#plt.scatter(X_reg_train, y_reg_train, c = 'b', label = 'Training Data')
#plt.scatter(X_reg_test, y_reg_test, c = 'g', label = 'Testing Data')
#plt.scatter(X_reg_test, y_reg_pred, c = 'r', label = 'Prediction Data')
#plt.legend()
#plt.show()

# The Missing peace: Non - linearity 

# Create compile and fit
model_4 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(1, activation = tf.keras.activations.linear)
])

model_4.compile(
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(lr=0.001),
    metrics = ['accuracy']
)

history_4 = model_4.fit(X, y, epochs = 100, verbose = 0)

print('\nMODEL 4 EVALUATION\n')
model_4.evaluate(X, y)

# Check out our data to remind ourselves what it looks like
#plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.RdYlBu)
#plt.show()

# Check the decision boundary for our latest model
#plot_decision_boundary(model_4, X, y)


# Lets try to build our first NN with a non linear activation function

model_5 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(1, activation = 'relu')
])

model_5.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(lr = 0.001),
    metrics = ['accuracy']
)

history_5 = model_5.fit(X, y, epochs = 100, verbose = 0)

print('\nMODEL 5 EVALUATION\n')
model_5.evaluate(X, y)

# Time to create a multilayer NN the same as Tensorflow Playground

model_6 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(4, activation = 'relu'),
    tf.keras.layers.Dense(4, activation = 'relu'),
    tf.keras.layers.Dense(1)
    
])

model_6.compile(
    loss = 'binary_crossentropy', 
    optimizer = tf.keras.optimizers.Adam(lr = 0.001),
    metrics = ['accuracy']
)

history_6 = model_6.fit(X, y, epochs = 100, verbose = 0)

# Lets evaluate the model
print('\nMODEL 6 EVALUATION\n')
model_6.evaluate(X, y)

# If we don't know what our models are doing, best to visuzalize
#plot_decision_boundary(model_6, X, y)


'------------------------------------------------------------'
'------------------------------------------------------------'
'------------------------------------------------------------'

# Lets try to incorporate an output activation function

model_7 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(4, activation = 'relu'),
    tf.keras.layers.Dense(4, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model_7.compile(
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)

history_7 = model_7.fit(X, y, epochs = 100, verbose = 0)

# Visualize
plot_decision_boundary(model_7, X, y)

print('\nMODEL 7 EVALUATION\n')
model_7.evaluate(X, y)

'''
Whats wrong w/ the predictions we've made? Are we really evaluating our model correctly?
Answer: No, we're not using training and testing datasets

Note: The combination of linear and non-linear functions is one of the key fundamentals of NN
'''