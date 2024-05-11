from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# Set up parameters
time_steps = 20
hidden_units = 2
epochs = 30

class attention(Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1),
                               initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
    x = Input(shape=input_shape)
    RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(RNN_layer)
    outputs = Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    model = Model(x,outputs)
    model.compile(loss='mse', optimizer='adam')
    return model

model_attention = create_RNN_with_attention(hidden_units=hidden_units, dense_units=1,
                                            input_shape=(time_steps,1), activation='tanh')
model_attention.summary()
