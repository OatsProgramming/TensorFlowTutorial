import tensorflow as tf
import pandas as pd
from os import system

system('clear')

# Saving our models allows us to use them outside of their training ground such as in web application
# or an app

# There are two ways to save our models:
#   SavedModel format (can be restored using tf.keras.models.load_model; also compatible with TensorFlow Serving)
#   HDF5 format

# set up
tf.random.set_seed(42)
X = tf.range(-100, 100, 4)
y = X + 10

# Training and Testing sets
X_train = X[:40]
X_test = X[40:]

y_train = y[:40]
y_test = y[40:]

# Model 1
model_1 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(1)
])

model_1.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

model_1.fit(tf.expand_dims(X_train, -1), y_train, epochs = 100, verbose = 0)
print('MODEL 1')

# Model 1 Prediction
y_pred_1 = model_1.predict(X_test)

# Model 1 MAE and MSE
mae_1 = tf.keras.losses.mae(y_test, tf.squeeze(y_pred_1))
mse_1 = tf.keras.losses.mse(y_test, tf.squeeze(y_pred_1))
print(mae_1, 'MAE 1')
print(mse_1, 'MSE 1')

'----------------------------------------------------------------------------'

# Model 2
model_2 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(100, input_shape = [1], name = 'input_layer'),
    tf.keras.layers.Dense(1, name = 'output_layer')
])

model_2.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

model_2.fit(tf.expand_dims(X, -1), y, epochs = 100, verbose = 0)
print('MODEL 2')

# Model 2 Prediction
y_pred_2 = model_2.predict(X_test)

# Model 2 MAE and MSE
mae_2 = tf.keras.losses.mae(y_test, tf.squeeze(y_pred_2))
mse_2 = tf.keras.losses.mse(y_test, tf.squeeze(y_pred_2))
print(mae_2, 'MAE 2')
print(mse_2, 'MSE 2')

'----------------------------------------------------------------------------'

# Model 3
model_3 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(100, input_shape = [1]),
    tf.keras.layers.Dense(1)
])

model_3.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

model_3.fit(tf.expand_dims(X_train, -1), y_train, epochs = 500, verbose = 0)
print('MODEL 3')

# Model 3 Prediction
y_pred_3 = model_3.predict(X_test)

# Model 3 MAE and MSE
mae_3 = tf.keras.losses.mae(y_test, tf.squeeze(y_pred_3))
mse_3 = tf.keras.losses.mse(y_test, tf.squeeze(y_pred_3))
print(mae_3, 'MAE 3')
print(mse_3, 'MSE 3')

'----------------------------------------------------------------------------'
# LESSON STARTS BELOW


# Save model using SavedModel format
# The savefile includes:
#   The model architecture, allowing to re-instatiate the model
#   The model weights
#   The state of the optimizer, allowing to resume training exactly where you left off
# Use this if you're staying in the TensorFlow environment
# After naming the filepath, it would be saved in the current working directory 
model_2.save(filepath = 'model_2_SAVEDMODEL_FORMAT')
print('MODEL 2 SAVED')

# Now let's try to save it in HDF5
#   A universal data format
#   Perfect for very large files
#   If using the model outside of the TensorFlow environment, best to use this one
# Use the '.h5' extension to indicate that the model should be saved to HDF5
model_3.save(filepath = 'model_3_HDF5.h5')
print('MODEL 3 SAVED')
