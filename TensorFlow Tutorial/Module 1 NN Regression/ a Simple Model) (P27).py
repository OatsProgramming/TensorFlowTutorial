import tensorflow as tf
import matplotlib.pyplot as plt
from os import system

system('clear')

'''
 Running experiments to improve our model
 Remember the TensorFlow workflow? (Review it if necessary)

 The ML practictioner's motto:
   Experiment, experiment, experiment

 With all that being said, how would we be able to lower the losses (increase accuracy) of our ML Algorithm?
    1)  Get more data - get more examples for your model to train on (more opportunities to learn patterns or
        relationships btwn features and labels)
    2)  Make your model larger (using a more complex model) - this might come in the form of more layer or more
        hidden units in each layer
    3)  Train for longer - give your model more chances to patterns in the data
'''

# Set up
tf.random.set_seed(42)
X = tf.range(-100, 100, 4)
y = X + 10

X_train = X[:40]
X_test = X[40:]

y_train = y[:40]
y_test = y[40:]

# With everything set up as it was before, we can see that the feature and label data is fixed
# That means we can either do the 2nd or 3rd option in trying to improve our model: increase layers/hidden units and training time

# Lets do 3 modelling experiments:
# Note: like the scientific method, tweak something one by one and see the results

# 'model_1' - same as the original model, 1 layer, trained for 100 epochs
# 'model_2' - 2 layers, trained for 100 epochs
# 'model_3' - 2 layers, trained for 500 epochs

model_1 = tf.keras.Sequential([ 
    tf.keras.layers.Dense(1)
])

model_1.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.SGD(),
    metrics = ['mae']
)

model_1.fit(tf.expand_dims(X_train, -1), y_train, epochs = 100)
print('MODEL 1')

# Make and plot predictions for model_1 to track our results
# Lets get outcome predictions
y_pred_1 = model_1.predict(X_test)
print(y_pred_1, 'OUTCOME PREDICTION 1')

# Now lets create a reusable function to plot our data for ease

def plot(
    train_feature,
    train_label,
    test_feature,
    test_label, 
    prediction
):
    # Set up the plot
    plt.figure(figsize = (10, 7))
    # Set the training data as blue
    plt.scatter(train_feature, train_label, c = 'b', label = 'Training Data')
    # Set the testing data as green
    plt.scatter(test_feature, test_label, c = 'g', label = 'Testing Data')
    # Set the prediction as red
    plt.scatter(test_feature, prediction, c = 'r', label = 'Prediction')
    # Set legend to avoid confusion
    plt.legend()
    # Visualize the plot
    plt.show()

# Now, lets plot the predictions of model_1
#plot(X_train, y_train, X_test, y_test, y_pred_1)

# Calculate model_1 evaluation metrics
mae_1 = tf.keras.losses.mae(y_test, tf.squeeze(y_pred_1))
mse_1 = tf.keras.losses.mse(y_test, tf.squeeze(y_pred_1))
print(mae_1, 'MAE 1')
print(mse_1, 'MSE 1')