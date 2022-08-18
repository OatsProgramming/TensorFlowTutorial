from turtle import shape
import tensorflow as tf
from os import system

system('clear')

# Squeezing a tensor (removing all single dimensions)

# Create a tensor
# We'll create a tensor w/ 50 elements and singular dimensions from the start
# Setting seed to get reproducible results
tf.random.set_seed(42)
g = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
print(g, 'TENSOR')
# Notice how there's multiple square brackets when printing

# Just to verify that it has multiple singular dimensions
print(g.shape, 'SHAPE')

# Now lets squeeze the tensor to remove all the singular dimensions
g_squeezed = tf.squeeze(g)
print(g_squeezed, 'SQUEEZED')
print(g_squeezed.shape, 'SHAPE AFTER SQUEEZE')