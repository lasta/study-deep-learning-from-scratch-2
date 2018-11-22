# -------------------------------------------------------------
# 01. utility fanctions
# -------------------------------------------------------------
# -*- coding: utf-8 -*-
import unicodedata
from functools import reduce
from operator import add
from typing import Dict, List

import numpy as np
from keras.layers import GRU, Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam


def append_eos(texts: List[str], eos: str) -> List[str]:
    """
    Appends EOS to each text.
    >>> texts = ["a\\n", "b\\n"]
    >>> eos = "<EOS>"
    >>> append_eos(texts=texts, eos=eos)
    ['a<EOS>', 'b<EOS>']
    """
    return [text.strip() + eos for text in texts]


def text_to_chars(text: str, normalizes=True) -> List[str]:
    """
    Splits text to chars list.

    :param text: to split
    :param normalizes: Unicode normalize if true.
    :return: character list.

    >>> text = "IKEA(ｲｹｱ)港北"
    >>> text_to_chars(text)
    ['I', 'K', 'E', 'A', '(', 'イ', 'ケ', 'ア', ')', '港', '北']
    """
    ignore_chars = ['']
    if normalizes:
        normalized_text = unicodedata.normalize("NFKC", text)
    else:
        normalized_text = text
    return [char for char in list(normalized_text) if char not in ignore_chars]


def calc_max_len(texts: List[str]) -> int:
    """
    Calculates max length of texts.
    :param texts: list of texts.
    :return: max lenght in texts.
    >>> texts = ["1", "1234", "123456789"]
    >>> calc_max_len(texts)
    9
    """
    return max(len(text) for text in texts)


def put_padding(text, max_length, filling_char="\t"):
    """
    Puts filling_char to text to align to max_length.
    :param text: text to be aligned
    :param max_length: length to align
    :param filling_char: charactor to align with.
    :return: text filled with :filling_char: aligns to max_length.
    >>> text = "1234"
    >>> max_length = 10
    >>> filling_char = "|"
    >>> put_padding(text, max_length, filling_char)
    '1234||||||'
    """
    text_len = len(text)
    if text_len >= max_length:
        return text
    filling_len = max_length - text_len
    filling_chars = filling_char * filling_len
    return text + filling_chars


def text_to_vec(text, term_to_vec):
    """
    Converts text to vector in corpus.
    When char in text is not in corpus, then raises KeyError.
    Facebook fastText processes terms are split with white space,
    so ignore whitespace in text.

    :param text: text to vector
    :param char_to_id: dictionary of character to id
    :return: vector (dim 1 of numpy array)
    >>> text = "IKEA"
    >>> term_to_vec = {'I': np.array([0]),\
                       'K': np.array([1]),\
                       'E': np.array([2]),\
                       'A': np.array([3]),\
                       '港': np.array([4]),\
                       '北': np.array([5]),\
                       '\\t': np.array([6])}
    >>> text_to_vec(text, term_to_vec)
    array([[0],
           [1],
           [2],
           [3]])

    >>> text = "IKE!" # "!" is not in char_to_id.
    >>> term_to_vec = {'I': np.array([0]),\
                       'K': np.array([1]),\
                       'E': np.array([2]),\
                       'A': np.array([3]),\
                       '港': np.array([4]),\
                       '北': np.array([5]),\
                       '\\t': np.array([6])}
    >>> text_to_vec(text, term_to_vec)
    array([[0],
           [1],
           [2],
           [0]])
    """
    chars = [char for char in list(text) if char != ' ']
    return np.array([char_to_vec(char, term_to_vec) for char in chars])


def char_to_vec(char, term_to_vec):
    """
    >>> char = 'a'
    >>> term_to_vec = {'a': np.array([[0], [1]])}
    >>> char_to_vec(char, term_to_vec)
    array([[0],
           [1]])

    >>> char = ' '
    >>> char_to_vec(char, term_to_vec)
    array([[0],
           [0]])
    """
    return term_to_vec.get(char, np.zeros_like(term_to_vec[list(term_to_vec)[0]]))


def convert_one_hot(corpus, vocabulary_size, max_length):
    """
    Converts to one-hot vectors.

    :param corpus: list of word ids (dim 1 of numpy array)
    :param vocabulary_size: vocabulary size
    :return: one-hot vectors (dim 2 of numpy array)
    >>> corpus = np.array([1, 2, 1, 0])
    >>> vocabulary_size = 5
    >>> max_length = 10
    >>> convert_one_hot(corpus, vocabulary_size, max_length)
    array([[0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=int32)

    >>> corpus = np.array([[1, 2, 1, 0], [1, 2, 3, 0]])
    >>> vocabulary_size = 5
    >>> max_length = 10
    >>> convert_one_hot(corpus, vocabulary_size, max_length)
    Traceback (most recent call last):
        ...
    ValueError: Corpus must be dim 1, but actual is 2

    >>> corpus = np.array([1, 2, 6])
    >>> vocabulary_size = 5
    >>> max_length = 10
    >>> convert_one_hot(corpus, vocabulary_size, max_length)
    Traceback (most recent call last):
        ...
    IndexError: index 6 is out of bounds for axis 1 with size 5
    """
    N = max_length

    if corpus.ndim != 1:
        raise ValueError(
            "Corpus must be dim 1, but actual is %d" % corpus.ndim)

    one_hot = np.zeros((N, vocabulary_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        one_hot[idx, word_id] = 1

    return one_hot


def train_data_to_train_and_target_vector(train_data):
    """
    Converts training data to training vectors (X) and target vectors (Y).
    >>> train_data = np.array([[0, 1, 0], [0, 0, 1]])
    >>> train_vectors, target_vectors = train_data_to_train_and_target_vector(train_data)
    >>> train_vectors
    [[array([0, 1, 0])]]
    >>> train_vectors[0][0]
    array([0, 1, 0])
    >>> target_vectors
    [array([0, 0, 1])]
    """
    train_vectors = [[vector] for vector in train_data][:-1]
    target_vectors = [vector for vector in train_data][1:]
    return train_vectors, target_vectors


def char_to_one_hot(char: str, char_to_id, vocabulary_size):
    """
    Convert char to one hot vector.
    >>> char = 'a'
    >>> char_to_id = {'a': 1, 'b': 2}
    >>> vocabulary_size = 5
    >>> char_to_one_hot(char, char_to_id, vocabulary_size)
    array([[0, 1, 0, 0, 0]], dtype=int32)
    """
    input_char_id = char_to_id[char]
    return convert_one_hot(np.array([input_char_id]), vocabulary_size, max_length=1)


def divide_list(l: List, chunk_num: int) -> List[List]:
    """
    Divides list into chunk_num lists.
    :param l: list to divide.
    :param chunk_num: divided list num.
    :return: divided lists.
    >>> l = list(range(1, 11))
    >>> chunk_num = 4
    >>> divide_list(l, chunk_num)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    """
    list_len = len(l)
    import math
    chunk_len = math.ceil(list_len / chunk_num)
    return [l[idx:idx + chunk_len] for idx in range(0, list_len, chunk_len)]


# ------------------------------------------------------------
# 08. Define training models and functions.
# ------------------------------------------------------------
def generate_model(max_length, input_dim, output_dim):
    """
    Generates RNN model.
    """
    print('Build model...')
    model = Sequential()
    model.add(GRU(128*20, return_sequences=False,
                  input_shape=(max_length, input_dim)))
    model.add(BatchNormalization())
    model.add(Dense(output_dim))
    # model.add(Activation('linear'))
    # model.add(Activation('sigmoid'))
    model.add(Activation('softmax'))
    optimizer = Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def build_model(train_vectors, target_vectors):
    trainer = np.array(train_vectors)
    target = np.array(target_vectors)
    # fix hyper parameters
    print("model = generate_model(1, %d, %d)" %
          (len(trainer), len(trainer[0][0])))
    model = generate_model(1, len(trainer[0][0]), len(target[0]))
    return model


def train(trainer, target, model, epochs=1):
    # fix hyper parameters
    model.fit(trainer, target, batch_size=128, epochs=epochs)
    return model


def predict(input_vector, model):
    return model.predict(np.array(input_vector))


# ------------------------------------------------------------
# 12. Define predition functions.
# ------------------------------------------------------------
def cos_sim(v1, v2):
    """
    Calculates cosine similarity.
    see: https://en.wikipedia.org/wiki/Cosine_similarity
    .. math::
        similarity = cos(theta) = (A • B) / (||A|| ||B||)

    :param v1: numpy array (dim 1), non zero vector
    :param v2: numpy array (dim 1), non zero vector
    :return: cosine similarity (-1.0 < similarity < 1.0)

    >>> v1 = np.array([1, 1, 1, 1])
    >>> v2 = np.array([1, 1, 1, 1])
    >>> cos_sim(v1, v2)
    1.0

    >>> v1 = np.array([1, 1, 1, 1])
    >>> v2 = np.array([-1, -1, -1, -1])
    >>> cos_sim(v1, v2)
    -1.0
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def max_cos_sim(vec, mat):
    """
    Find max of cosine similarity to :vec: in :mat:
    :param vec: numpy array (dim 1), non zero vector
    :param mat: list of numpy array (dim 1), non zero vectors
    :return: vector of max of cosine similarity im :mat:

    >>> vec = np.array([1, 1, 1, 1])
    >>> mat = [np.array([1, 1, 1, 1]), np.array([-1, -1, -1, -1])]
    >>> max_cos_sim(vec, mat)
    array([1, 1, 1, 1])
    """
    return max(mat, key=lambda col: cos_sim(col, vec))


def find_nearest(arr, val):
    """
    Finds nearest :val: in :arr:
    """
    idx = np.abs(np.asarray(arr) - val).argmin()
    return arr[idx]


def find_char_by_vec(vec: np.ndarray, term_vec: Dict[str, List[float]]) -> str:
    """
    Finds character in :term_vec: equals to :vec:
    :vec: numpy array
    :term_vec: dictionary of character to vector (scalar array).
    :return: character of vector, or None.
    >>> vec = np.array([0, 0, 1])
    >>> term_vec = {'a': [0, 0, 0], 'b': [0, 0, 1]}
    >>> find_char_by_vec(vec, term_vec)
    'b'
    >>> vec_not_in_term_vec = np.array([0, 0, -1])
    >>> find_char_by_vec(vec_not_in_term_vec, term_vec)
    """
    chars = [key for key, value in term_vec.items() if value == vec.tolist()]
    if chars:
        return chars[0]


def fill_with_eos(text, length, eos):
    """
    Fills with eos to resize to length.
    :param text: to fill with eos.
    :param length: max length
    :param eos: to fill with, must be length of 1.
    
    >>> text = "1234"
    >>> length = 10
    >>> eos = "0"
    >>> fill_with_eos(text, length, eos)
    '1234000000'
    """
    if len(text) >= length:
        return text
    filling_length = length - len(text)
    return text + eos * filling_length
