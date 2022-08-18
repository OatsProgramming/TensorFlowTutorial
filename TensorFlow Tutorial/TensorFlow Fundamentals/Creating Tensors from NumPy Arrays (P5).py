import tensorflow as tf
import numpy as np
import os

os.system('clear')

# If we want our shuffled tensors to be in the same order, 
# we've got to use the global and operation level random seed

# To make reproducible data, we would want to shuffle our data in a similar order
# and initialize with similiar random weights

# Create a tensor of all ones
print(tf.ones([10,7]))

# Create a tensor of all zeroes
print(tf.zeros(shape=(3,4)))

# You can also turn NumPy arrays into tensors
# The main difference btwn NumPy arrays and TensorFlow tensor is that tensors can be run
# on a GPU computing
numpy_a = np.arange(1, 25, dtype=np.int32) # Create a NumPy array btwn 1 and 25
print(numpy_a, 'NUMPY A')
# x = tf.constant(some_matrix) Captial for matrix/tensor
# y = tf.constant(vector) Noncapital for vector

A = tf.constant(numpy_a)
print(A, 'NUMPY INTO TENSOR')
B = tf.constant(numpy_a, shape=(2, 3, 4)) # Shape must correspond to list size to work
print(B, 'TENSOR DIFF SHAPE')