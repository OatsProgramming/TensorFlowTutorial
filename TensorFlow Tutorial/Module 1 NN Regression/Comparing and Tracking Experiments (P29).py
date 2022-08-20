import tensorflow as tf
import numpy as np
import pandas as pd
from os import system

system('clear')


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



# After running a few experiments, let's compare the results
# Note: You want to start w/ small experiments (models) and make sure they work and then increase their scale when necessary

# Let's compare our model's results using a pandas DataFrame (its built in tensorflow)

model_results = [ 
    ['model_1', mae_1.numpy(), mse_1.numpy()],
    ['model_2', mae_2.numpy(), mse_2.numpy()],
    ['model_3', mae_3.numpy(), mse_3.numpy()]
]

all_results = pd.DataFrame(model_results, columns=['model', 'mae', 'mse'])
print(all_results)

# Looks like model_1 performed best...
# We still need to find a way to increase the accuracy (lower loss) and time efficiency

# Tracking your experiments:
#   One good habit in ML modelling is to track the results of your experiments
#   When doing so, it can be tedious. Luckily, there are tools that can help with that
#   Look into the following:
#       TensorBoard
#       Weights & Biases - a tool for tracking all kinds of ML experiments (plugs straight into TensorBoard)