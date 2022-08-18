import tensorflow as tf
from os import system

system('clear')

x = tf.constant([[1, 2], [3, 4], [5, 6]])
y = tf.constant([[7, 8], [9, 10], [11, 12]])
# In order to see if this will multiply, lets go over the rules and see
# Check the inner dimensions if they match
print(x.shape, 'X SHAPE') # (3, 2)
print(y.shape, 'Y SHAPE') # (3, 2)
# Inner dimensions (x * y) ---> (x(3x2) * y(3x2)) ---> x(2) y(3)
# This does not work bc the inner dimensions do not match

# (3, 2)
a = tf.constant([[1, 2], 
                [3, 4], 
                [5, 6]])
# (2, 3)
b = tf.constant([[7, 8, 9],  
                [10, 11, 12]])
# Now lets try again
# The inner dimensions (a * b) ---> (a(3x2) * b(2x3)) ---> a(2) b(2) ---> Valid
# Therefore, we can multiply these matrices
print(tf.matmul(a, b), 'PROPER MATRIX MULTIPLICATION')
# In the second rule, the resulting matrix's size is the same as the outer dimensions
# Therefore, the shape of the resulting matrix is (3, 3)

# We can also change the shape 
y = tf.reshape(y, (2, 3))
print(y.shape, 'Y SHAPE CHANGED')
print(y, 'Y ELEMENTS AFTER RESHAPED')

# Now we can also multiply x tensor and y tensor
print(tf.matmul(x, y), 'XY MATRICES MULTIPLIED')

# You can get something similar to transpose
print(x, 'BEFORE TRANSPOSE')
print(tf.transpose(x), 'AFTER TRANSPOSE')
print(tf.reshape(x, (2, 3)), 'RESHAPE INSTEAD')
# After printing the result, take a closer look at the elements
# In transpose, it takes the first dimension and set it as the second dimension (vice versa),
# which ultimately results to:
# [[1, 3, 5], 
# [2, 4, 6]]
# Whereas reshaping it keeps the elements as is but moves the [] placement
# [[1, 2, 3],
# [4, 5, 6]]
# This would greatly affect the outcome when multiplied

# Note: find examples of matrix multiplication being used
