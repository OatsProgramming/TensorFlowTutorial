import tensorflow as tf
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




# We can recreate a saved model, including its weights and optimizer by using
# tf.keras.models.load_model('enter file/folder name')

# Lets load in the SavedModel format model (Be sure to check to path)
loaded_SavedModel_format = tf.keras.models.load_model('model_2_SAVEDMODEL_FORMAT')

# Lets see if it works by comparing
loaded_SavedModel_format.summary()
print('SAVEDMODEL SUMMARY')
model_2.summary()
print('MODEL 2 SUMMARY')

# Now let's check if the weights are the same by checking the predictions
SavedModel_pred = loaded_SavedModel_format.predict(X_test)
print(SavedModel_pred, 'SAVEDMODEL PREDICTION')
print(y_pred_2, 'MODEL 2 PREDICTION')
print(SavedModel_pred == y_pred_2, 'CHECK FOR EQUALITY')

# It returns False
# So what if we check to see if MAE are the same?
SavedModel_mae = tf.keras.losses.mae(y_test, tf.squeeze(SavedModel_pred))
print(SavedModel_mae, 'SAVEDMODEL MAE')
print(mae_2, 'MODEL 2 MAE')
print(SavedModel_mae == mae_2, 'CHECK FOR MAE EQUALITY')

model3_copy = tf.keras.models.load_model('model_3_HDF5.h5')

model3_pred = model3_copy.predict(X_test)
print(y_pred_3, 'PREDICTION 3')
print(model3_pred, 'PREDICTION COPY')
print(y_pred_3 == model3_pred)

model3_copy_mae = tf.keras.losses.mae(y_test, tf.squeeze(model3_pred))
print(model3_copy_mae == mae_3)