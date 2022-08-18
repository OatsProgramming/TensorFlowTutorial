import tensorflow as tf
import os

os.system('clear')

# Lets shuffle the order of elements in a tensor

# Lets say we were making an image detector and we give 10_000 Ramen pics and 5_000 Spaghetti pics
# Depending on the order, it can affect the NN to only focus on identifying Ramen or Spaghetti
# However, we would like to identify both. That's why itd be best to shuffle the pics with no order at all

# Shuffle a tensor (So the inherent order doesnt affect learning)
not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [2,5]])
print(not_shuffled, 'UNSHUFFLED')

# Randomly shuffles a tensor along its first dimension
# e.g. shape=(3,2) Focuses on the '3' i.e the rows
# Pretty much shuffles the rows
shuffled = tf.random.shuffle(not_shuffled)
print(shuffled, 'SHUFFLED')

# What happens if we were to set the seed?
tf.random.set_seed(42) # global seed
shuffled = tf.random.shuffle(not_shuffled, seed = 42) # operation-level seed
print(shuffled, 'SHUFFLE WITH SEED')

# Operations that rely on a random seed actually derive it from two seeds:
# The global and operation-level seeds

# Refer to TensorFlow Doc (TensorFlow random seed)

