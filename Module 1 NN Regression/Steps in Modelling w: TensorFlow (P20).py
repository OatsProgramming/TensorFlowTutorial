from pickletools import optimize
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import system

system('clear')

X = np.array([-7., -4., -1., 2., 5., 8., 11., 14.])
y = np.array([3., 6., 9., 12., 15., 18., 21., 24.])

input_shape = X[0].shape
output_shape = y[0].shape

print(input_shape, 'NP SHAPE IN')
print(output_shape, 'NP SHAPE OUT')

# Turn our NumPy arrays into tensors
X = tf.constant(X)
y = tf.constant(y)
input_shape = X[0].shape
output_shape = y[0].shape
print(input_shape, 'TENSOR SHAPE IN')
print(output_shape, 'TENSOR SHAPE OUT')

# Notice how the shapes are 0; this is due to the fact that the values we're attaining are
# scalar shaped. We want to get one input value that results in one output value

# How might we create a model to determine the relationship btwn X and y?

'''
Steps in modelling w/ TensorFlow
    1) Creating a model
        Define the input and output layers, as well as, the hidden layers of a deep learning model
    2) Compiling a model
        Define the loss function (in other words, the function which tells our model how wrong it is)
        and the optimizer (tells our model how to improve the patterns its learning) and evaluation
        metrics (what we can use to interpret the performance of our model)
    3) Fitting a model
        Letting the model try to patterns btwn X and y
'''

# Set random seed
tf.random.set_seed(42)

# 1) Create a model using the Sequential API
model = tf.keras.Sequential([       # Create a model and sequentially go thru the following
    tf.keras.layers.Dense(1)        # We use '1' bc we're inputting just one
])

# You can also add layers by doing the following:
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1))

# 2) Compile the model
model.compile(
    loss=tf.keras.losses.mae, # mae: mean absolute error: Predicted v Observed; loss = mean(abs(y_true - y_pred), axis=-1)
    optimizer=tf.keras.optimizers.SGD(), # sgd: stochastic gradient descent
    metrics=['mae']
)

# 3) Fit the model
# Note: for TensorFlow 2.7.0+, fit() no longer upscales input data to go from (batch_size, ) to
# (batch_size, 1). To fix this, you'll need to expand the dimensions of input data using:
# tf.expand_dims(input_data, axis=-1)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=5) # Look at X and y to find the relationship with just 5 tries

# After running the model, notice the speed. What if we made it faster by changing the dtype to something lower?

X = tf.cast(X, dtype=tf.float32)
y = tf.cast(y, tf.float32)

model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(1))

model2.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['mae']
)
print("")
print("")
print("")
print('WITH FLOAT32')
model2.fit(tf.expand_dims(X, -1), y, epochs=5)

# Try and make a prediction using our model
print(model.predict([17.0]), 'PREDICTION') # Thats the input/feature/x; its going to try and predict the output/label/y

# We can evaluate our model by using .evaluate()
print(model.evaluate(X, y))

# The prediction doesnt go too well. We'll see ways to improve it on the next lesson
