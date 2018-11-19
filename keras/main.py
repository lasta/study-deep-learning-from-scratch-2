# -*- coding: utf-8 -*-
import numpy as np
import sys
import codecs
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
from preprocess import preprocess, text_to_vec, convert_one_hot, calc_max_len
from functools import reduce
from operator import add

# End of string of training texts.
EOS = '\t'

# 431 rows
raw_train_data_path = './data/names_gt_1000.tsv'
# 178362 rows
# raw_train_data_path = './data/names_has_uu.tsv'

with codecs.open(raw_train_data_path, 'r', 'utf-8') as file:
    train_texts = [text.strip() + EOS for text in file.readlines()]

corpus, char_to_id, id_to_char = preprocess(train_texts, padding_char=EOS)
vocabulary_size = len(char_to_id)

combined_texts = reduce(add, train_texts)
max_length = len(combined_texts)
print("max_length : %d" % max_length)
text_vectors = text_to_vec(combined_texts, char_to_id)
text_one_hot_vectors = convert_one_hot(text_vectors, vocabulary_size, max_length)
print("len(text_one_hot_vectors) : %d" % len(text_one_hot_vectors))
print("len(text_one_hot_vectors[0]) : %d" % len(text_one_hot_vectors[0]))


# TODO: convert train_vec and label_vec (embedding_matrix)
"""
train_vec -> label_vec
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

* それぞれで同じ長さのベクトル列 or 行列を作る?
"""
train_vectors = [[vector[:-1]] for vector in text_one_hot_vectors]
target_vectors = [vector[1:] for vector in text_one_hot_vectors]

# print(type(combined_train_vectors))
# print(combined_train_vectors[1])

def generate_model(max_length, input_dim, output_dim):
    """
    Generates LSTM model.
    """
    print('Build model...')
    model = Sequential()
    # TODO: 128? 20?
    model.add(GRU(128*20, return_sequences=False, input_shape=(max_length, input_dim)))
    model.add(BatchNormalization())
    model.add(Dense(output_dim))
    model.add(Activation('linear'))
    model.add(Activation('sigmoid'))
    # model.add(Activation('softmax'))
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


def predict(input_char, model):
    # input_vector = convert_one_hot(np.array([char_to_id[input_char]]), vocabulary_size, max_length)
    input_char_id = char_to_id[input_char]
    # TODO: max_length を動的に扱う
    input_vector = convert_one_hot(np.array([input_char_id]), vocabulary_size, max_length=1)
    input_vector = np.array([[vector[1:]] for vector in input_vector ])

    output_vector = model.predict(np.array(input_vector))
    max_idx = output_vector.argmax()
    return id_to_char[max_idx]


model = train(train_vectors, target_vectors)
predicted_char = predict('I', model)

print("predicted : " + predicted_char)
