from holdem import Table, PlayerControl
from qlearning.train import Train

from keras.layers import *
from keras.models import Sequential
from keras.optimizers import *

seats = 10
grid_size = 66
hidden_size = 32
nb_frames = 5
lstm_size = 64
batch_size = 5000

model = Sequential()
# model.add(Convolution1D(32, 3, activation='relu', input_shape=(nb_frames, grid_size)))
# model.add(Convolution1D(32, 3, activation='relu'))
# model.add(Convolution1D(16, 3, activation='relu'))
# model.add(Flatten())
model.add(LSTM(lstm_size, return_sequences=True, input_shape=(nb_frames, grid_size)))
model.add(Dropout(0.2))
model.add(LSTM(int(lstm_size / 2), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(int(lstm_size / 4)))
# model.add(Dense(int(hidden_size), activation='relu', input_shape=(nb_frames, grid_size)))
# model.add(Dropout(0.2))
# model.add(Dense(int(hidden_size/2), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(int(hidden_size/4), activation='relu'))
# model.add(Dense(6))
model.compile(RMSprop(), 'MSE')
model._make_predict_function()

# controller for human meat bag
# h = PlayerControl(1, t)

trainers = []
for i in range(1, seats + 1):
    trainers.append(Train())
    # pp = PlayerControlProxy(p)

epoch = 0
while True:
    t = Table(seats, training=True, quiet=True)
    print('starting ai players')
    # fill the rest of the table with ai players
    for i in range(1, seats + 1):
        p = PlayerControl(i, t, model=model, train=trainers[i-1])
    t.run_game()
    epoch += 1