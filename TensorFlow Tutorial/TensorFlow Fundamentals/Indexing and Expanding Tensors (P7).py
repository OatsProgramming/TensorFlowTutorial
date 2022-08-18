import tensorflow as tf
import os

os.system('clear')

# Tensors can be indexed just like Python Lists

# Get the first two elements of each dimension
rank_4_tensors = tf.zeros(shape=(2, 3, 4, 5))
print(rank_4_tensors[:2, :2, :2, :2])

# Get the first element from each dimension from each index except for the final one
print(rank_4_tensors[:1, :1, :1, :], 'SHAPE 1')

# Check the changes in the shape
a = rank_4_tensors[:1, :1, :, :1]
b = rank_4_tensors[:1, :, :1, :1]
c = rank_4_tensors[:, :1, :1, :1]

print(a, 'SHAPE 2')
print(b, 'SHAPE 3')
print(c, 'SHAPE 4')

# Create a rank 2 tensor (2 dimensions)
rank_2_tensor = tf.constant([[10, 7],
                        [3, 4]])
print(rank_2_tensor.ndim, 'RANK 2')

# Get the last item of each row of our rank 2 tensor
print(rank_2_tensor[:, -1], 'LAST ITEMS OF RANK 2')

# Add in an extra dimension to our rank 2 tensor
rank_3_tensor = rank_2_tensor[..., tf.newaxis] # [every axis before, tf.newaxis]
print(rank_3_tensor, 'ADDING A DIMENSION TO RANK 2')

# Alternative to tf.newaxis
tf.expand_dims(rank_2_tensor, axis=-1)

# You can also expand on the 0 axis
tf.expand_dims(rank_2_tensor, axis=0)

