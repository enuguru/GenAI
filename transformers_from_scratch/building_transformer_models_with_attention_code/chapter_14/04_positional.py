import tensorflow as tf
from tensorflow.keras.layers import Embedding

output_length = 6
output_sequence_length = 5

position_embedding_layer = Embedding(output_sequence_length, output_length)
position_indices = tf.range(output_sequence_length)
embedded_indices = position_embedding_layer(position_indices)
print(embedded_indices)
