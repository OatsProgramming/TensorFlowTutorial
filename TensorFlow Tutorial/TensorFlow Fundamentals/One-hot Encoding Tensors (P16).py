import tensorflow as tf
from os import system

system('clear')

'''
What is one-hot encoding?
    A form of numerical encoding (Turning inputs to data)

    Red     Green   Blue
   [1       0       0],
   [0       1       0],
   [0       0       1]

    Row 2 has 1 for 'red' and 0 for the rest; so, that is encoded for red
    Row 3 has 1 for 'green' and 0 for 'red' and 'blue'; so, that is encoded for green
    Row 4 has 1 for 'blue' and 0 for 'red' and 'green'; so that is encoded for blue

    One-hot encoded for red: [1, 0, 0]
    One-hot encoded for green: [0, 1, 0]
    One-hot encoded for blue: [0, 0, 1]
'''

# Create a list of indices
some_list = [0, 1, 2, 3] # Could be red, green, blue, purple

# One hot encode our list of indices
# To one hot encode, we must give it a depth relative to the amnt of elements
# so that it can encode all of it
# Depth = column
print(tf.one_hot(some_list, depth=4), 'ONE HOT')

'''
[[1. 0. 0. 0.]  For 0
 [0. 1. 0. 0.]  For 1
 [0. 0. 1. 0.]  For 2
 [0. 0. 0. 1.]] For 3
'''

# This would be rare but
# We can specify custom values for one hot encoding
print(tf.one_hot(some_list, depth=4, on_value='Whaddup G', off_value="See ya bro"))