import tensorflow as tf
import os

os.system('clear')

# Main Tensor attributes:
    # Shape: 
    #   the length (number of elements) of each of the dimensions of a tensor
    #   tensor.shape
    # Rank:
    #   The number of tensor dimensions. A scalar has rank 0, vector 1, matrix 2 and Tensor n
    #   tensor.ndim
    # Axis or Dimension
    #   A particular dimension of a tensor
    #   tensor[0], tensor[:, 1]...
    # Size
    #   The total number of items in the tensor
    #   tf.size(tensor)

# Create a rank 4 tensor
# Tip: 
    # Within the tensor, there are 2 groups
    # Within the the 2 groups there are 3 subgroups per each
    # Within the 3 subgroups, there are 4 rows
    # With the 4 rows, there are 5 columns
rank_4_tensors = tf.zeros(shape=[2, 3, 4, 5])
print(rank_4_tensors)

# Getting the elements along a certain dimension
print(rank_4_tensors[0], 'FIRST DIMENSION')
print(rank_4_tensors[1], 'SECOND DIMENSION')
# And so on...

print(rank_4_tensors.shape, 'SHAPE') # Should be the same as previous
print(rank_4_tensors.ndim, 'NDIM') # 4 (Total elements/dimensions in shape)
print(tf.size(rank_4_tensors), 'SIZE') # 120 (Elements multiplied together)

