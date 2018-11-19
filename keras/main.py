# -*- coding: utf-8 -*-
import numpy as np
import sys
# from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, SimpleRNN
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise as GN
from keras.layers.noise import GaussianDropout as GD
import tensorflow as tf
from utils import preprocess, text_to_vec, convert_one_hot, calc_max_len, open_raw_data, append_EOS, train_data_to_train_and_target_vector, char_to_one_hot
from functools import reduce
from operator import add

# End of string of training texts.
EOS: str = '\t'

# 431 rows
INPUT_FILE_PATH = './data/names_gt_1000.tsv'
# 178362 rows
# raw_train_data_path = './data/names_has_uu.tsv'

def generate_model(max_length, input_dim, output_dim):
    """
    Generates LSTM model.
    """
    print('Build model...')
    model = Sequential()
    model.add(GRU(128*20, return_sequences=False, input_shape=(max_length, input_dim)))
    model.add(BatchNormalization())
    model.add(Dense(output_dim))
    # model.add(Activation('linear'))
    # model.add(Activation('sigmoid'))
    model.add(Activation('softmax'))
    optimizer = Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def train(train_vectors, target_vectors, epochs=1):
    trainer = np.array(train_vectors)
    target = np.array(target_vectors)
    # TODO: fix hyper parameters
    print("model = generate_model(1, %d, %d)" % (len(trainer), len(trainer[0][0])))
    model = generate_model(1, len(trainer[0][0]), len(target[0]))
    # TODO: fix hyper parameters
    model.fit(trainer, target, batch_size=128, epochs=epochs)
    return model


def predict(input_vector, model):
    return  model.predict(np.array(input_vector))


# -- main --
raw_data = open_raw_data(INPUT_FILE_PATH)
train_texts = append_EOS(raw_data, eos=EOS)

corpus, char_to_id, id_to_char = preprocess(train_texts, padding_char=EOS)
vocabulary_size = len(char_to_id)

combined_texts = reduce(add, train_texts)
max_length = len(combined_texts)
text_vectors = text_to_vec(combined_texts, char_to_id)
text_one_hot_vectors = convert_one_hot(text_vectors, vocabulary_size, max_length)

"""
train_vec -> target_vec
[0]: 東 -> 京
[1]: 京 -> タ
[2]: タ -> ワ
[3]: ワ -> ー
[4]: ー -> <EOS>
[5]: <EOS> -> 0 (padding)
[max_length + 1]: I -> K
[max_length + 2]: K -> E
[max_length + 3]: E -> A
[max_length + 4]: A -> 立
[max_length + 5]: 立 -> 川
[max_length + 6]: 川 -> <EOS>
[max_length + 7]: <EOS> -> 0 (padding)
"""
train_vectors, target_vectors = train_data_to_train_and_target_vector(text_one_hot_vectors)

model = train(train_vectors, target_vectors)

# predict
input_char = 'I'
input_one_hot_vector = char_to_one_hot(input_char, char_to_id, vocabulary_size)
# reshape to input dim matrix
input_vector = np.array([[vector[1:]] for vector in input_one_hot_vector])

output_vector = predict(input_vector, model)

max_idx = output_vector.argmax()
predicted_char = id_to_char[max_idx]
print("predicted : " + predicted_char)
