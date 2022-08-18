import tensorflow as tf
import numpy as np
from os import system


system('clear')

# Tensors and NumPy
# TensorFlows interacts beautifully w/ NumPy arrays

# Create a tensor directly from a NumPy array
a = tf.constant(np.array([3., 7., 10.]))
print(a, 'TENSOR')

# Convert our tensor back to a NumPy array
print(np.array(a), 'NUMPY ARRAY')
print(type(np.array(a)), 'TYPE')

# Convert the tensor into a NumPy array
print(a.numpy(), 'CONVERTED NUMPY')

# The default types of each are slightly different
numpy_a = tf.constant(np.array([3., 7., 10.]))
tensor_a = tf.constant([3., 7., 10.])

# Check the datatypes of each
print(numpy_a.dtype, 'NUMPY DTYPE') # Default dtype: float64
print(tensor_a.dtype, 'TENSOR DTYPE') # Default dtype: float 32