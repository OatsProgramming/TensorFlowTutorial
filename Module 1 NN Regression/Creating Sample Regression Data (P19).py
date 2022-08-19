import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import system

system('clear')

# There are many different def for a regression problem but in our case, we're going to simplify it:
# predicting a number based on other numbers

# Creating data to view and fit

# Create features
X = np.array([-7., -4., -1., 2., 5., 8., 11., 14.])

# Create labels
y = np.array([3., 6., 9., 12., 15., 18., 21., 24.])

# Visualize it
# plt.scatter(X, y)
# plt.show()

# Lets figure out the relationship btwn X and y 
print(y == X + 10, 'RELATIONSHIP')

# Input and output shapes

# Create a demo tensor for our housing price prediction problem
house_info = tf.constant(['bedroom', 'bathroom', 'garage'])
house_price = tf.constant([939_700])
print(house_info, 'HOUSE INFO') # Shape would be (3, )
print(house_price, 'HOUSE PRICE') # Shape would be (1, )

input_shape = X.shape
output_shape = y.shape
