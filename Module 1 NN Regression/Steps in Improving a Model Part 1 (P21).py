import tensorflow as tf
import numpy as np
from os import system

system('clear')

'''
Steps in Modelling w/ TensorFlow

    1) Construct or import a pretrained model relevant to the problem
    2) Compile the model (prepare it to be used w/ data)
        Loss:
            How wrong the model's predictions are compared to the truth labels (Needs to be constantly minimized)
        Optimizer:
            How your model should update its internal patterns to better its predictions
        Metrics:
            Human interpretable values for how well your model is doing
    3) Fit the model to the training data so it can discover patterns
        Epochs:
            How many times the model will go thru all of the training examples
    4) Evaluate the model on the test data (how reliable are the model's predictions)
'''

# Create tensors that our model can read
X = tf.constant(np.array([-7., -4., -1., 2., 5., 8., 11., 14.]))
y = tf.constant(np.array([3., 6., 9., 12., 15., 18., 21., 24.]))

# Set the seed for reproducible results
tf.random.set_seed(42)

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
])

# Compile the model
model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

# Fit the model
model.fit(tf.expand_dims(X, axis = -1), y, epochs = 5)

print(model.predict([17.0]))

# With all that set up, how might we improve the model?

# We can improve our model by altering the steps we took to create a model
# 1) Creating a model:
#       Add more layers? Increase neurons? Change the activation?
# 2) Compiling a model:
#       Change the optimization? Or the learning rate of the optimization function?
# 3) Fitting a model:
#       More training? More data?
