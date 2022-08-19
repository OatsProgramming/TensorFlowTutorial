import tensorflow as tf
from keras.utils import plot_model
from os import system

system('clear')

# Before starting, let's get our set up just like the previous lesson

# Set seed for reproducibility
tf.random.set_seed(42)

# Make some tensors
X = tf.range(-100, 100, 4)  # Features
y = X + 10                  # Labels

# Get training (80% for this example) and testing (20% for this example) sets from the data
X_train = X[:40]
X_test = X[40:]

y_train = y[:40]
y_test = y[40:]

# Create a model
model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(10, input_shape = [1], name = 'input_layer_WOOT'), # We'll use 10 Neurons instead of 1
    tf.keras.layers.Dense(1, name = 'output_layer_POOT')
], name = 'MODEL_NAME_WOOTAPOOT')

# Compile
model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    metrics = ['mae']
)

# Fit
model.fit(tf.expand_dims(X_train, -1), y_train, epochs = 100, verbose = 0)

# Get a summary of our model
# Take note as to how the names have been changed in the summary
model.summary()

# Another way to visualize our model is to use the plot model function
plot_model(
    model, 
    to_file = 'model.png',      # Note: This will pop up on your current directory (i.e. Module 1 NN)
    show_shapes = True,
    show_dtype = False
)

# A lot of the time, we'll spend w/ our NN making sure that the inputs and outputs are corrrect
