# -*- coding: utf-8 -*-
import re
import numpy as np
from typing import List


def append_EOS(texts: List[str], eos: str) -> List[str]:
    """
    Appends EOS to each text.
    >>> texts = ["a\\n", "b\\n"]
    >>> eos = "<EOS>"
    >>> append_EOS(texts=texts, eos=eos)
    ['a<EOS>', 'b<EOS>']
    """
    return [text.strip() + eos for text in texts]


def preprocess(texts, puts_padding=False, padding_char="\t"):
    """
    Preprocessor for text to char vector
    :param texts: raw corpus
    :param puts_padding: pad with padding_char to align text length.
    :return: corpus: corpus
             char_to_id: char in corpus to id map
             id_to_char: id to char in corpus map

    >>> texts = ['IKEA港北', 'IKEA新三郷']
    >>> corpus, char_to_id, id_to_char = preprocess(texts)
    >>> corpus
    array([1, 2, 3, 4, 7, 8, 9])
    >>> char_to_id
    {'\\t': 0, 'I': 1, 'K': 2, 'E': 3, 'A': 4, '港': 5, '北': 6, '新': 7, '三': 8, '郷': 9}
    >>> id_to_char
    {0: '\\t', 1: 'I', 2: 'K', 3: 'E', 4: 'A', 5: '港', 6: '北', 7: '新', 8: '三', 9: '郷'}

    >>> corpus, char_to_id, id_to_char = preprocess(texts, puts_padding=True)
    >>> corpus
    array([1, 2, 3, 4, 7, 8, 9])
    >>> char_to_id
    {'\\t': 0, 'I': 1, 'K': 2, 'E': 3, 'A': 4, '港': 5, '北': 6, '新': 7, '三': 8, '郷': 9}
    >>> id_to_char
    {0: '\\t', 1: 'I', 2: 'K', 3: 'E', 4: 'A', 5: '港', 6: '北', 7: '新', 8: '三', 9: '郷'}
    """
    if puts_padding:
        max_text_len = calc_max_len(texts)
        texts = [put_padding(text, max_text_len, padding_char) for text in texts]
    chars_list = [text_to_chars(text) for text in texts]

    # Initialize with padding char.
    char_to_id = {padding_char: 0}
    id_to_char = {0 : padding_char}

    for chars in chars_list:
        for char in chars:
            if char not in char_to_id:
                new_id = len(char_to_id)
                char_to_id[char] = new_id
                id_to_char[new_id] = char
    
    corpus = np.array([char_to_id[char] for char in chars])

    return corpus, char_to_id, id_to_char


def text_to_chars(text):
    """
    Splits text to chars list.

    >>> text = "IKEA港北"
    >>> text_to_chars(text)
    ['I', 'K', 'E', 'A', '港', '北']
    """
    ignore_chars = ['']
    # TODO: normalize Japanese.
    return [char for char in list(text) if char not in ignore_chars]


def calc_max_len(texts):
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
    >>> term_to_vec = {'I': np.array([0]), 'K': np.array([1]), 'E': np.array([2]), 'A': np.array([3]), '港': np.array([4]), '北': np.array([5]), '\\t': np.array([6])}
    >>> text_to_vec(text, term_to_vec)
    array([[0],
           [1],
           [2],
           [3]])

    >>> text = "IKE!" # "!" is not in char_to_id.
    >>> term_to_vec = {'I': np.array([0]), 'K': np.array([1]), 'E': np.array([2]), 'A': np.array([3]), '港': np.array([4]), '北': np.array([5]), '\\t': np.array([6])}
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
        raise ValueError("Corpus must be dim 1, but actual is %d" % corpus.ndim)

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
    [[array([0, 1])], [array([0, 0])]]
    >>> train_vectors[0][0]
    array([0, 1])
    >>> target_vectors
    [array([1, 0]), array([0, 1])]
    """
    train_vectors = [[vector[:-1]] for vector in train_data]
    target_vectors = [vector[1:] for vector in train_data]
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


def generate_npy(texts, char_to_id, output_file):
    """
    >>> texts = ["IKEA港北", "IKEA三郷", "IKEA立川"]
    >>> _, char_to_id, _ = preprocess(texts)
    >>> output_file = "/tmp/generate_npy.npy"
    >>> generate_npy(texts, char_to_id, output_file)
    array([[ 1,  2,  3,  4,  5,  6],
           [ 1,  2,  3,  4,  7,  8],
           [ 1,  2,  3,  4,  9, 10]])
    """
    # TODO: text の長さが違っていてもできるようにする
    # もしかしたら、np.arrayのpython.listでもよいのかもしれない
    # corpus : [array([0, 1, 2, 3, 4, 5]), array([0, 1, 2, 3, 6, 7]), array([0, 1, 2, 3, 8, 9])]
    texts_vec = [np.array([char_to_id[char] for char in text]).reshape(len(text)) for text in texts]
    corpus = np.array(texts_vec)
    np.save(output_file, corpus)
    return corpus