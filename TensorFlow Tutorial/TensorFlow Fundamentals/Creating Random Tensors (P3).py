from random import random
import tensorflow as tf
import os

os.system('clear')

# Rarely in practice will you need to decide whether to use tf.constant or tf.Variable
# to create tensors, as TensorFlow does this for you. However, if in doubt, use tf.constant
# and change it later if needed.

# Random tensors are tensors of some arbitrary size which contain random numbers

# Where does the NN get the representations from 
# 1) Initializes with random weights (only at beginning)
# 2) Show examples
# 3) Update representation outputs
# 4) Repeat w/ more examples

# Think of it a wave oscillating back and forth (Random Tensors) 
# till it lines up to the proper answer

random_1 = tf.random.Generator.from_seed(42) # set seed for reproducibility
# What is tensorflow random normal?
# Outputs random values from a normal distribution
# What is a normal distribution?
# A function that represents the distributions of many random variables as a symmetrical bell-shaped graph
random_1 = random_1.normal(shape=(3,2))
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3,2))
# This would produce a tensor (Matrix)
print(random_1, 'RANDOM 1')
print(random_2, 'RANDOM 2')

# Are the equal?
print(random_1 == random_2, 'EQUAL')
# Yes they do

# The numbers appear to be random but they are sudo random due to setting the seed
