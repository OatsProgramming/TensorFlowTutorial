import tensorflow as tf
import os

os.system('clear')

# print(tf.__version__) # This is self explanatory (Last checked: 2.9.1)

# Create Tensors with tf.constant()
# Shape has 0 elements
scalar = tf.constant(7)
print(scalar, 'scalar')

# Check the number of dimensions of a tensor (ndim: Number of Dimensions)
print(scalar.ndim, 'scalar.ndim')

# Create a vector
# Shape has 1 element
vector = tf.constant([10,10])
print(vector, 'vector')

# Check the dimension of our vector
print(vector.ndim, 'vector.ndim')

# Create a matrix (Has more than 1 dimension)
# Shape has 2 elements
matrix = tf.constant([[10,7],
                    [7, 10]])
print(matrix, 'matrix')

# Check the dimension of our matrix
print(matrix.ndim, 'matrix.ndim')

# Create another matrix but this time, mess with data type
# By default, we get our data type as 32int (32 bit precision)
# The higher number of precision, the more exact the numbers below are 
# Shape is 3 by 2 (3 rows, 2 columns)
another_matrix = tf.constant([[10., 7.],
                            [3., 2.],
                            [8., 9.]], dtype=tf.float16) # Specify the data type with dtype paramenter 
print(another_matrix, 'another matrix')

# After checking out the the ndims of the previous datas, what would
# the ndim for 'another matrix' be?
print(another_matrix.ndim, 'another matrix ndim')
# It would be 2: the ndim correlates to the number of elements in the shape


# Lets create a tensor
tensor = tf.constant([[[1, 2, 3], 
                        [4, 5, 6]],

                        [[7, 8, 9],
                        [10, 11, 12]],

                        [[13, 14, 15],
                        [16, 17, 18]]])
print(tensor, 'tensor')

print(tensor.ndim, 'tensor ndim')

# Thus far we know this:
# A scalar has 0 dimensions
# A vector (a number with direction) has 1 dimension
# A matrix has 2 dimensions
# A tensor (an n-dimensional array of numbers)(when n can be an number)
