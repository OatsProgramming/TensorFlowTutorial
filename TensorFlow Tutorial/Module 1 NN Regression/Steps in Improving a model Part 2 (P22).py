import tensorflow as tf
import numpy as np
from os import system

system('clear')

# Before starting, set up the data and model

# Set up the tensors
X = tf.constant(np.array([-7., -4., -1., 2., 5., 8., 11., 14.]))
y = tf.constant(np.array([3., 6., 9., 12., 15., 18., 21., 24.]))

# Set the seed for reproducible results
tf.random.set_seed(42)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

# Fit
model.fit(tf.expand_dims(X, axis = -1), y, epochs = 5)

# Before improving, we want to create a smaller model first and see how we can improve from there (intuitive)
# As of rn, we only have: one layer, one neuron, trains abt 5 times, and uses a small subset data instead of using the entire thing
# Also, the optimizer we're using can be changed

# Let's rebuild our model by first increasing the layers and the neurons
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

print("")
print("")
print("")
print('INCREASING LAYERS AND NEURONS')
model.fit(tf.expand_dims(X, axis = -1), y, epochs = 5)

# By printing this out, you will see that loss has decreased in comparison to the previous model
# This ultimately makes the model more accurate; however, we want the loss to be as low as possible
# So what if we increased the training laps?

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

print("")
print("")
print("")
print('INCREASED LAYERS, NEURONS, AND EPOCHS')
model.fit(tf.expand_dims(X, -1), y, epochs = 100)

# By increasing the training time, we also decreased the loss; however, for some reason, once it reaches a certain lvl 
# of loss/mae, it starts to oscillate, never reaching below 1
# So what if we changed the optimizer?

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)
])  

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr=0.0001), # lr: Learning Rate
    metrics = ['mae']
)

print("")
print("")
print("")
print("CHANGED OPTIMIZERS")
model.fit(tf.expand_dims(X, -1), y, epochs = 100)

# By changing the optimizer and setting the learning rate, we can see that in longer oscillates
# Every time we try to improve the model, it's best to scientifically approach this: using the scientific method and change one thing at a time

# For the sake of improving our model, what if we added an activation in the layer?

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr=0.0001),
    metrics = ['mae']
)

print("")
print("")
print("")
print("CHANGED ACTIVATION AND LESSEN LAYERS")
model.fit(tf.expand_dims(X, -1), y, epochs = 100)

# ReLU: 
#   Commonly used activation function in deep learning
#   Returns 0 if the input is negative, but for any positive input, it returns that value back


'''
Common ways to improve the model
    Adding layers
    Increase the number of hidden units
    Change the activation function
    Change the optimization function
    Change the learning rate (Potentially the most important parameter of any ML)
    Increasing data
    Training for longer
'''