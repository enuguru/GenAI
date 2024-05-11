from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential

# Set up parameters
time_steps = 20
hidden_units = 2
epochs = 30

# Create a traditional RNN network
def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mse', optimizer='adam')
    return model

model_RNN = create_RNN(hidden_units=hidden_units, dense_units=1,
                       input_shape=(time_steps,1), activation=['tanh', 'tanh'])
model_RNN.summary()
