import tensorflow as tf
from os import system

system('clear')

# In machine learning, matrix multiplication is the most common operation

# There are two rules our tensors (or matrices) need to fulfill if we're going to matrix
# multiply them:
# 1) The inner dimensions must match
# 2) The resulting matrix has the shape of the outer dimensions

# To multiply a matrix by another matrix we need to do the 'dot product' of rows and columns
# Ex:
# First workout the answer for 1st row and the 1st column
#   [1, 2, 3    [7, 8                   [58
#    4, 5, 6]    9, 10         --->             ]
#                11, 12]
# The Dot Product is where we multiply matching members, then sum up:
# (1, 2, 3) * (7, 9, 11) ---> (1*7) + (2*9) + (3*11) ---> 58
# Repeat for the 1st row and 2nd Column
# Then 2nd row, 1st column
# Then 2nd row, 2nd column
# And so on...
# Refer to matrixmultiplication.xyz website for more details

# Matrix multiplication in Tensorflow
tensor = tf.constant([[10, 7], [3, 4]])
print(tf.matmul(tensor, tensor), 'MATRICES MULTIPLIED')

# To multiply with Python operation
print(tensor @ tensor, 'MATRICES MULTIPLIED WITH PYTHON')

# Create a tensor (3, 2)
x = tf.constant([[1, 2], [3, 4], [5, 6]])
y = tf.constant([[7, 8], [9, 10], [11, 12]])
# Why would multiplying these tensors not work?
# Check for part 2
