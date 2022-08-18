import os
import tensorflow as tf

os.system('clear')

# Create the same tensor with tf.variable() as prev lesson
changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10,7])
print(unchangeable_tensor, 'VARIABLE')
print(changeable_tensor, 'CONSTANT')

# Lets try to change one of the elements in our changeable tensor
# changeable_tensor[0]
print(changeable_tensor[0]) # This would return the first value in the list
# changeable_tensor[0] = 7
# This would raise a TypeError

# How abt we try .assign()
changeable_tensor[0].assign(7)
print(changeable_tensor, 'ELEMENT CHANGED')

# Now let's try to change our unchangeable tensor
# unchangeable_tensor[0] = 7
# Would raise a similar error as to before
# unchangeable_tensor[0].assign(7)
# This would raise an AttributeError

