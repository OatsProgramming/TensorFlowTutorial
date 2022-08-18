import tensorflow as tf
from os import system

system('clear')

'''
 When would finding the positional Min and Max of a tensor be helpful?
   In the Ramen v Spaghetti example, when NN outputs probabilities, 
   we have to distinct the two thru Mins and Maxes

                   [[0.983, 0.004, 0.013],
                    [0.110, 0.889, 0.001],
                    [0.023, 0.027, 0.985], .......

    For this example, let's work with these data sets (The Representation Outputs). 
    And for now, we're going to pretend that column(index) 0: Ramen; column(index) 1: Spaghetti; column(index) 2: Neither
                        INDEX
                  0         1         2
                Ramen   Spaghetti   Neither
                   [[0.983, 0.004, 0.013],
                    [0.110, 0.889, 0.001],
                    [0.023, 0.027, 0.985], .......

    Take note as to how in each column there is a max and min
    We want to get the the positional max and min via Column x Row
    So basically, at which index of the tensor/row does the max/min value occur
 Note: Update representation outputs is often referred to as "Prediction Probabilities"
'''

# Create a new tensor for finding positional min and max
# We'll use tf.random.uniform()
# This will output random numbers based on it's shape, min value, max value, etc.
# Setting seed to get reproducible results
tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])
print(F, 'TENSOR')

# Lets find the positional max
print(tf.argmax(F), 'POSITIONAL MAX') # Results in index 42

# Index on our largest value position
print(F[tf.argmax(F)], 'LARGEST VALUE POSITION') # Results in 0.9671384

# Find the max value
print(tf.reduce_max(F), 'MAX VALUE') # Results in 0.9671384

# Check for equality
print(F[tf.argmax(F)] == tf.reduce_max(F)) # Returns True



# Lets find the positional min
print(tf.argmin(F), 'POSITIONAL MIN') # Results in index 16

# Find the minimum value by using the index
print(F[tf.argmin(F)], 'SMALLEST VALUE POSITION') # Results in 0.009463668

# Find the min value
print(tf.reduce_min(F), 'MIN VALUE') # Results in 0.009463668

# Check for equality
print(F[tf.argmin(F)] == tf.reduce_min(F)) # Returns True

