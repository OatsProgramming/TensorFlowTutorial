import tensorflow as tf
import matplotlib.pyplot as plt
from os import system

system('clear')

# Just like the exercise in the previous lesson, we're going to create a model and visualize it
# We've already done model_1, so now lets do model_2 and model_3
# 'model_2' - 2 layers, 100 hidden units, trained for 100 epochs
# 'model_3' - 2 layers, 100 hidden units, trained for 500 epochs

# Create a plot function for ease
def plot(
    train_feature,
    train_label,
    test_feature,
    test_label,
    prediction,
    model_number
):
    # Set up plot
    plt.figure(figsize = (10, 7))
    # Set title to create distinction btwn other models
    plt.title(model_number)
    # Set training data as blue
    plt.scatter(train_feature, train_label, c = 'b', label = 'Training Data')
    # Set testing data as green
    plt.scatter(test_feature, test_label, c = 'g', label = 'Testing Data')
    # Set prediction data as red
    plt.scatter(test_feature, prediction, c = 'r', label = 'Prediction')
    # Set legend to avoid confusion
    plt.legend()
    # Visualize the plot
    plt.show()



# Set up

# Set seed for reproducibility
tf.random.set_seed(42)

# Get data (features and labels, respectively)
X = tf.range(-100, 100, 4)
y = X + 10

# Make Training and Testing sets
X_train = X[:40]
X_test = X[40:]

y_train = y[:40]
y_test = y[40:]

# Model up
model_2 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(100, input_shape = [1], name = 'input_layer'),
    tf.keras.layers.Dense(1, name = 'output_layer')
])

model_2.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

model_2.fit(tf.expand_dims(X, -1), y, epochs = 100)
print('MODEL 2')

# Get prediction data
y_pred_2 = model_2.predict(X_test)

# Now let's see it on the plot
plot(X_train, y_train, X_test, y_test, y_pred_2, model_number='MODEL 2')

# How abt the evaluation metrics of model_2?
# Reminder: we're squeezing y_pred_2 bc we need the both of them to be of the same shape
mae_2 = tf.keras.losses.mae(y_test, tf.squeeze(y_pred_2))
mse_2 = tf.keras.losses.mse(y_test, tf.squeeze(y_pred_2))
print(mae_2, 'MAE 2')
print(mse_2, 'MSE 2')



# Now let's try with model_3 and see how it differs

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

# Get the predictions of model_3
y_pred_3 = model_3.predict(X_test)

# Show in the plot
plot(X_train, y_train, X_test, y_test, y_pred_3, model_number = 'MODEL 3')

# What abt the evaluation metrics?
mae_3 = tf.keras.losses.mae(y_test, tf.squeeze(y_pred_3))
mse_3 = tf.keras.losses.mse(y_test, tf.squeeze(y_pred_3))
print(mae_3, 'MAE 3')
print(mse_3, 'MSE 3')




'''
After running the three models, you would notice how it starts to worsen
This is a prime example of 'overfitting'
Think of it as overtraining or studying too much in one sitting; this ultimately leads to a burn out
In machine learning, we need to find the goldilucks zone while also making the algorithm as efficient as possible

"   Overfitting happens when a model learns the detail and noise in the training data to the extent that it 
    negatively impacts the performance of the model on new data. This means that the noise or random fluctuations 
    in the training data is picked up and learned as concepts by the model "
'''