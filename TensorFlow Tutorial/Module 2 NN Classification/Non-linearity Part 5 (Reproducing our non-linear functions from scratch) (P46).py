import tensorflow as tf
import matplotlib.pyplot as plt
from os import system

system('clear')

'------------------------------------------------------------'
'------------------------------------------------------------'
'------------------------------------------------------------'

'''
Now, we've discussed the concept of linear and nonlinear lines, how abt we see them
in action?
'''

# Create a toy tensor (similar to the data we pass into our models)
A = tf.cast(tf.range(-10, 10), tf.float32)

# Visualize our toy tensor
#plt.plot(A)
#plt.show()

# Lets start by replicating sigmoid
# tf.exp == exponent
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

# Lets use the sigmoid function in our toy tensor
print(sigmoid(A))

# Plot our toy tensor transformed by sigmoid
#plt.plot(sigmoid(A))
#plt.show()

# This will turn A linear to nonlinear



# Now that we know what sigmoid does, what abt relu?
# Relu:
#   In a NN, the activation function is responsible for transforming the summed weighted
#   input from the node into the activation of the node or output for that input
#   Essentially: max(x, 0)

# Lets recreate the relu function
def relu(x):
    return tf.maximum(0, x)

# Lets pass our toy tensor to our custom relu function
print(relu(A))

# Based on what we're seeing, it just turns any negative number as 0

# Lets see it on a plot
#plt.plot(relu(A))
#plt.show()



# Lets try the linear activation function
tf.keras.activations.linear(A)

# Based on what we're seeing, it doesnt modify anything
# To verify
print('\nA == LINEAR A?\n')
print(A == tf.keras.activations.linear(A))

# Lets visualize it
#plt.plot(tf.keras.activations.linear(A))
#plt.show()


# For more in depth stuff of activation functions:
#   Machine Learning Cheat Sheet