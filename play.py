from holdem import Table, PlayerControl
from qlearning.train import Train

from keras.layers import *
from keras.models import Sequential
from keras.optimizers import *

seats = 10
grid_size = 66
hidden_size = 32
nb_frames = 5
lstm_size = 16
batch_size = 5000
learning_rate = 0.00001

model = Sequential()
# model.add(Convolution1D(int(hidden_size), 3, activation='relu', input_shape=(nb_frames, grid_size)))
# model.add(Dropout(0.2))
model.add(Dense(int(hidden_size), activation='relu', input_shape=(nb_frames, grid_size)))
model.add(Dropout(0.2))
model.add(Dense(int(hidden_size/2), activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(int(lstm_size)))

model.compile(Adam(lr=learning_rate), 'MSE')
model._make_predict_function()

# controller for human meat bag
# h = PlayerControl(1, t)

trainers = []
for i in range(1, seats + 1):
    trainers.append(Train())

while True:
    t = Table(seats, training=True, quiet=True)
    print('starting ai players')
    # fill the rest of the table with ai players
    for i in range(1, seats + 1):
        p = PlayerControl(i, t, model=model, train=trainers[i-1])
    t.run_game()