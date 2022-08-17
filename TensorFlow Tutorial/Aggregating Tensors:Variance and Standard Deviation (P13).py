import tensorflow as tf
import numpy as np
from os import system

system('clear')

# Aggregating tensors:
#   Condensing them from multiple values to a smaller amnt of values

# Get the absolute values by using tf.abs()
a = tf.constant([-7, -10])
print(tf.abs(a), 'ABSOLUTE VALUE')

# Lets go thru the following forms of aggregation:
#   Min, Max, Mean, Sum

# Create a random tensor with values btwn 0 and 100 of size 50
b = tf.constant(np.random.randint(0, 100, size = 50))
print(b, 'B TENSOR')
print(tf.size(b), 'B SIZE') # Try to remember that its tf.size() not b.size()
print(b.shape, 'B SHAPE')
print(b.ndim, 'B RANK')
print(b.dtype, 'B DATATYPE')

# Find the minimum
print(tf.reduce_min(b), 'B MINIMUM')

# Find the max
print(tf.reduce_max(b), 'B MAX')

# Find the mean
print(tf.reduce_mean(b), "B MEAN")

# Find the sum
print(tf.reduce_sum(b), "B SUM")

# Variance:
#   tf.math.reduce_variance()
#   Calculated by taking the avg of squared deviations from the mean
#   Tells you the degree of spread in your data set
#   The more spread the data, the larger the variance is in relation to the mean
#
#                    ∑(xi - x̅)**2
#    s**2    =       --------------
#                       (n - 1)
# s**2: Sample Variance
# xi: the value of the one observation
# x̅: the mean value of all observation
# n: the number of observations

# Standard Deviation:
#   tf.reduce_std()
#   A measure of the amount of variation or dispersion of a set of values
#                                             ______________
#                                            / ∑(xi - µ)**2
#   Population Standard Deviation =      \  / -------------
#                                         \/       N
# N: the size of the population
# xi: each value from the population
# µ: population mean

# In order to find the variance and/or standard deviation of a tensor,
# the data must first be converted to a float
b = tf.cast(b, dtype = tf.float32)

# Now, we can proceed

# Find the variance of b tensor
print(tf.math.reduce_variance(b), 'B VARIANCE')

# Find the standard deviation of b tensor
print(tf.math.reduce_std(b), 'B STANDARD DEVIATION')