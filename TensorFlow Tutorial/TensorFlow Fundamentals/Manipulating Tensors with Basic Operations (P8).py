import tensorflow as tf
import os 

os.system('clear')

# Basic Operations
# +, -, *, /

# You can add values to a tensor
tensor = tf.constant([[10, 7], [3, 4]])
print(tensor + 10)

# Multiply
print(tensor * 10)

# Subtraction
print(tensor - 10)

# And so on...

# We can use the tensorflow built-in function too
print(tf.multiply(tensor, 10))

# Although they yield the same result, using the tensorflow operations instead of basic ones
# help speed up the process as it uses the GPU
# When applicable, use tensorflow operations

# Refer to the doc