import tensorflow as tf
from os import system

system('clear')

# Most datatypes are in int32
# However, sometimes we want to change that
# We can use tf.cast()

# Create a new tensor w/ default datatype (float32)
a = tf.constant([1.7, 7.4])
b = tf.constant([1, 2])
print(a.dtype, 'A DATATYPE') # This would be float32 since theres floats inside
print(b.dtype, 'B DATATYPE') # This would be int32 since its just integers

# Change from float32 to float16 (This is called: REDUCED PRECISION)
# Mixed Precision:
#   The use of both 16-bit and 32-bit floating point types in a model during a training
#   to make it run faster and use less memory
# There are two lower precision dtypes: float16 and bfloat16
c = tf.cast(b, dtype = tf.float16)
print(c.dtype, 'REDUCED PRECISION')

# Change float32 to int32
d = tf.cast(a, dtype = tf.int32)
print(d.dtype, 'FLOAT TO INT')

# Change int32 to float16
e = tf.cast(b, dtype = tf.float16)
print(e.dtype, 'INT32 TO FLOAT16')

