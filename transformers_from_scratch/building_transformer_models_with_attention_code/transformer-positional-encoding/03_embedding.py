import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow.keras.layers import TextVectorization, Embedding
from tensorflow.data import Dataset

output_length = 6
output_sequence_length = 5
vocab_size = 10
sentences = [["I am a robot"], ["you too robot"]]

sentence_data = Dataset.from_tensor_slices(sentences)
# Create the TextVectorization layer
vectorize_layer = TextVectorization(output_sequence_length=output_sequence_length,
                                    max_tokens=vocab_size)
# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)
# Convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)
# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)

word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words = word_embedding_layer(vectorized_words)
print(embedded_words)
