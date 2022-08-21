import tensorflow as tf
from sklearn.datasets import make_circles
from os import system

system('clear')

# Create circles
X, y = make_circles(
    n_samples = 1000,
    noise = 0.03,
    random_state = 42
)

# Set seed
tf.random.set_seed(42)

# Create, compile, fit, evaluate

model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(1)
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    metrics = ['accuracy']
)

model.fit(X, y, epochs = 5, verbose = 0)

print('EVALUATION 1')
model.evaluate(X, y)

'------------------------------------------------------'

# Lets try to improve our model

model_2 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_2.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    metrics = ['accuracy']
)

model_2.fit(X, y, epochs = 100, verbose = 0)

print('EVALUATION 2')
model_2.evaluate(X, y)

# Although we've increased the layers and epochs, we're getting poorer results
# What should we visualize in order to improve our model?
