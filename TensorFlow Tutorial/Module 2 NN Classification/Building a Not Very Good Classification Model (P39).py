import tensorflow as tf
from os import system
from sklearn.datasets import make_circles

system('clear')

# Sample size
n_samples = 1000

# Create circles
X, y = make_circles(
    n_samples,
    noise = 0.03,
    random_state = 42
)

'------------------------'

# Create a model

# Set seed for reproducibility
tf.random.set_seed(42)

# Create, compile, fit, evaluate
model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(1)
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    metrics = ['accuracy'] # Its in percent (e.g. 0.48 == 48%)
)

model.fit(X, y, epochs = 5)

print("EVALUATION")
model.evaluate(X, y)