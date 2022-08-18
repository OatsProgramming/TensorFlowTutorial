import tensorflow as tf
from os import system

system('clear')

# The dot product
# Matrix multiplication is also referred to as the dot product
# You can perform matrix multiplication tf.matmul() or tf.tensordot() or @
# Usually, just use tf.matmul()

# Perform the dot product on x and y (requires x or y to be transposed)
x = tf.constant([[1, 2], 
                [3, 4], 
                [5, 6]])
y = tf.constant([[7, 8], 
                [9, 10], 
                [11, 12]])

print(tf.tensordot(tf.transpose(x), y, axes = 1), 'DOT PRODUCT')

# Perform matrix multiplication btwn x and y (transposed)
print(tf.matmul(x, tf.transpose(y)), 'TRANSPOSED')

# Perform matrix multiplication btwn x and y (reshape)
print(tf.matmul(x, tf.reshape(y, (2, 3))), 'RESHAPED')

# Check the values of Y, reshaped y, and transposed y
print("NORMAL Y:")
print(y)
print("Y RESHAPED:")
print(tf.reshape(y, (2, 3)))
print("Y TRANSPOSED")
print(tf.transpose(y))

# Most of the time, reshaping and transposing would be done behind the scene
# Generally, when perforoming matrix multiplication on two tensors and one of the 
# axes doesnt line up, you will transpose (rather than reshape) one of the tensors to
# satisfy the matrix multiplication rules.

