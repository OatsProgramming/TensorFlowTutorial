import tensorflow as tf
from os import system

system('clear')

# Squaring, log, and square root

# Create a new tensor
# We can also create a new tensor by implementing tf.range()
# Similar to python's range
a = tf.range(1, 10)
print(a, 'TENSOR')

# Square it
print(tf.square(a), 'SQUARED')

# Square root
# In order to square root our tensor, since we're square rooting,
# we must change it to an acceptable datatype that would have decimals
# For our case, we'll just change it to a datatype float32
a_float = tf.cast(a, tf.float32)
print(tf.sqrt(a_float), 'SQUARE ROOT')

# Find the log
# Remember, it will have decimals
print(tf.math.log(a_float), 'LOGGED')
