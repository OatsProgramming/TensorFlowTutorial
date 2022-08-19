import tensorflow as tf
import matplotlib.pyplot as plt
from os import system

system('clear')

# Evaluating a model
# In practice, a typical workflow you'll go thru when building a NN is:
#   Build a model ---> fit it ---> evaluate it ---> tweake a model ---> fit it ---> evalutate it ---> repeat till desire result

# When it comes to evalution... there are 3 words to memorize: VISUALIZE VISUALIZE VISUALIZE
# Its a good idea to visualize:
#   The data (What data are we working w/? What does it look like?)
#   The mode itself (What does our model look like?)
#   The training of a model (How does a model perform while it learns?)
#   The predictions of a model (How do the predictions of a model line up against the actual outcomes?)


# Lets make a big dataset
X = tf.range(-100, 100, 4)
print(X, 'DATASET')

# Make labels for the dataset
# The following is the pattern that we want the Algorithm to learn
y = X + 10 
print(y, 'LABELS')

# Visualize the data
# plt.scatter(X, y)
# plt.show()

# Before getting to getting more in depth in visualizing, lets familiarize ourselves with the 3 sets
#   Training Set    (Model learns from this data, which is typically 70-80% of the total data you have available)
#   Validation Set  (Model gets tuned on this data, which is typically 10-15% of the data available)
#   Test Set        (Model gets evaluated on this data to test what it has learned; typically 10-15% of the total data available)

# Analogy:
#   Training Set    --->    Course materials 
#   Validation Set  --->    Practice exams
#   Test Set        --->    Final exams

# Check the length of how many samples we have
print(len(X), 'LENGTH') # Would return 50

# Split the data into train and test sets
X_train = X[:40] # We want the first 40 for training (80% of the data)
y_train = y[:40]

X_test = X[40:] # We want the last 10 for testing (20% of the data)
y_test = y[40:]

print(len(X_train), len(X_test), 'X TRAIN AND TEST')
print(len(y_train), len(y_test), 'Y TRAIN AND TEST')

# Visualizing the data
#   Now we've got our data in training and test sets... let's visualize it again

plt.figure(figsize=(10, 7))
# Plot training data in blue 'b'
plt.scatter(X_train, y_train, c='b', label='Training data') # Our model will learn from this
# Plot testing data in green 'g'
plt.scatter(X_test, y_test, c='g', label='Testing data') # We want our model to be able to predict this (given x, what's y?)
# Show a legend for distinction purposes
plt.legend()
# Visualize
# plt.show()

# We want the algorithm to learn from the training data then eventually test itself with the testing data

# Lets have a look at how to build a NN for our data

# Create a model
model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    metrics = ['mae']
)

# # Fit the model (We'll fit the training data)
# model.fit(tf.expand_dims(X_train, -1), y_train, epochs = 100)

# Visualizing the model
#   We can get a glimpse as to what our model looks like by doing the following:
# model.summary()
# We need the input shape defined first

# Lets create a model which builds automatically by defining the input shape argument in the first layer
# Set seed for reproducibility
tf.random.set_seed(42)

# Check the input shape
print(X[0])

# Create a model (Same as above) but with the input_shape
model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(1, input_shape = [1]) # Since we're just passing one number, input_shape is 1
])

# Compile the model (Also the same)
model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr=0.01),
    metrics = ['mae']
)

# Now, since we got the input shape, we can check the model summary
model.summary()

# Non-trainable params:
#   These parameters aren't updated during training (this is typical when you bring in already learned patterns
#   or parameters from other models during transfer learning)

# What are the trainable params in a NN?

# For a more in depth overview of the trainable params w/in a layer, check out MIT's intro to Deep Learning Course

# EXERCISE: try playing around w/ the number of hidden units in the dense layer, see how that effeects the # of params
# (total and trainable) by calling 'model.summary()'

# Lets fit the model but set the verbose
# We can either set to 0, 1, or 2:
#   0 is no progress bar (no training illustrated)
#   1 is to show the training
#   2 is to show the training w/o progress bar (recommended when not running interactively)
model.fit(tf.expand_dims(X, -1), y, epochs = 100, verbose = 1)