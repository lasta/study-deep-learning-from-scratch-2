import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation


def sin2p(x, t=100):
    # cycle
    return np.sin(2.0 * np.pi * x / t)

def sin_data(t=100, cycle=2):
    x = np.arange(0, cycle * t)
    return sin2p(x, t)

def noisy(Y, noise_range=(-0.05, 0.05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise


# prepare data
np.random.seed(0)

raw_data = noisy(sin_data(100, 2), (-0.05, 0.05))
input_len = 20


input = []
target = []

for i in range(0, len(raw_data) - input_len):
    input.append(raw_data[i:i + input_len])
    target.append(raw_data[i + input_len])

X = np.array(input).reshape(len(input), input_len, 1)
Y = np.array(target).reshape(len(input), 1)
x, val_x, y, val_y = train_test_split(X, Y, test_size=int(len(X) * 0.2), shuffle=False)


# train
n_in = 1
n_hidden = 20
n_out = 1
epochs = 10
batch_size = 10

model = Sequential()
model.add(SimpleRNN(n_hidden, input_shape=(input_len, n_in), kernel_initializer='random_normal'))
model.add(Dense(n_out, kernel_initializer='random_normal'))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999))
model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))


# predict
in_ = x[:1]
predicted = [None for _ in range(input_len)]
for _ in range(len(raw_data) - input_len):
    out_ = model.predict(in_)
    in_ = np.concatenate((in_.reshape(input_len, n_in)[1:], out_), axis=0).reshape(1, input_len, n_in)

    predicted.append(out_.reshape(-1))

plt.title('predict sin wave')
plt.plot(raw_data, label='original')
plt.plot(predicted, label='predicted')
plt.plot(x[0], label='input')
plt.legend()
plt.show()