import tensorflow as tf
from keras.utils import plot_model
import matplotlib.pyplot as plt
from os import system

system('clear')

# Visualizing our model's predictions:
#   To visualize predictions, its a good idea to plot them against the ground truth (the actual outcome) labels
#   Often, you'll see this: 'y_true' or 'y_test' vs. 'y_pred'

# Set up
tf.random.set_seed(42)

X = tf.range(-100, 100, 4)
y = X + 10

X_train = X[:40]
X_test = X[40:]

y_train = y[:40]
y_test = y[40:]

model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(10, input_shape = [1], name = 'input_layer_WOOT'),
    tf.keras.layers.Dense(1, name = 'output_layer_POOT')
], name = 'model_name_WOOTAPOOT')

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr=0.01),
    metrics = ['mae']
)

model.fit(tf.expand_dims(X_train, -1), y_train, epochs = 100, verbose = 0)

# Make some predictions
y_pred = model.predict(X_test) # Predict on the X test data set so that we can compare the y test data set
print(y_pred, 'OUTCOME PREDICTIONS')
print(y_test, 'OUTCOME ACTUAL')

# Lets compare it by creating a plotting function
def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions
):
    plt.figure(figsize = (10, 7))
    # Plot training in blue
    plt.scatter(train_data, train_labels, c = 'b', label = 'Training data')
    # Plot testing in green
    plt.scatter(test_data, test_labels, c = 'g', label = 'Testing Data')
    # Plot model's predictions in red
    plt.scatter(test_data, predictions, c = 'r', label = 'Predictions')
    # Show the legend
    plt.legend()
    # Show the plot
    plt.show()

plot_predictions(X_train, y_train, X_test, y_test, y_pred)
# The objective is to get the red line as close as possible to the green line


