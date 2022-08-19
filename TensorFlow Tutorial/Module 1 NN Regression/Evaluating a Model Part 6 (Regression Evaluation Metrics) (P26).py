import tensorflow as tf
from os import system

system('clear')

# Evaluating our model's predictions w/ regression evaluation metrics
#   Depending on the problem you're working on, there will be different evaluation metrics
#   to evaluate your model's performance.
#   Since we're working on a regression, two of the main metrics:
#       MAE - Mean Absolute Error; 'On avg, how wrong is each of my model's predictions'
#       MSE - Mean Square Error; 'Square the avg errors'

'''
In calculating the MAE:

    ∑(top n; bottom i=1)|yi - xi|
    -----------------------------
                n

    1) Find the absolute difference btwn predicted value and the actual value
    2) Sum all these values
    3) Find their avg

    Notes:
        yi: label
        xi: feature

    When to use:
        Great starter metric for any regression problem

MSE:

    (1/n) * ∑(top n; bottom i=1) * (yi - ŷi)**2

    Notes:
        yi: label
        ŷi: y predictions
    
    When to use:
        When larger errors are more significant than smaller errors

Huber:
    Unknown
'''

# Set everything up

# Set seed for reproducibility
tf.random.set_seed(42)

# Get some tensors
X = tf.range(-100, 100, 4)
y = X + 10

# Training and Testing sets
X_train = X[:40]
X_test = X[40:]

y_train = y[:40]
y_test = y[40:]

# Create, Compile, and fit model
model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(100, input_shape = [1], name = 'input_layer'),
    tf.keras.layers.Dense(1, name = 'output_layer')
])

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr=0.01),
    metrics = ['mae']
)

model.fit(tf.expand_dims(X, -1), y, epochs = 100, verbose = 0)

# Now get predictions
y_pred = model.predict(X_test)



# Now, evaluate the model on the test set
model.evaluate(X_test, y_test)

# Calculate the mean absolute error (MAE)
# Use the other one (not tf.keras.losses.mae) for practice

# If we ever want to compare, we got to make sure that they're both the same shape
print(y_test.shape, 'Y TEST SHAPE')
print(y_pred.shape, 'Y PREDICT SHAPE')

# They both differ in shape: 
#   y_test's shape: (10, )
#   y_pred's shape: (10, 1)
# A way we can solve this is by removing all the one dimensions of y_pred
print(tf.metrics.mean_absolute_error(y_test, tf.squeeze(y_pred)), 'MAE')




# Calculate the mean square error (MSE)
print(tf.metrics.mean_squared_error(y_test, tf.squeeze(y_pred)), 'MSE')