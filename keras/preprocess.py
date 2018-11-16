# -*- coding: utf-8 -*-
import re
import numpy as np


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
    array([0, 1, 2, 3, 6, 7, 8])
    >>> char_to_id
    {'I': 0, 'K': 1, 'E': 2, 'A': 3, '港': 4, '北': 5, '新': 6, '三': 7, '郷': 8}
    >>> id_to_char
    {0: 'I', 1: 'K', 2: 'E', 3: 'A', 4: '港', 5: '北', 6: '新', 7: '三', 8: '郷'}

    >>> corpus, char_to_id, id_to_char = preprocess(texts, puts_padding=True)
    >>> corpus
    array([0, 1, 2, 3, 7, 8, 9])
    >>> char_to_id
    {'I': 0, 'K': 1, 'E': 2, 'A': 3, '港': 4, '北': 5, '\\t': 6, '新': 7, '三': 8, '郷': 9}
    >>> id_to_char
    {0: 'I', 1: 'K', 2: 'E', 3: 'A', 4: '港', 5: '北', 6: '\\t', 7: '新', 8: '三', 9: '郷'}
    """
    if puts_padding:
        max_text_len = calc_max_len(texts)
        texts = [put_padding(text, max_text_len, padding_char) for text in texts]
    chars_list = [text_to_chars(text) for text in texts]

    char_to_id = {}
    id_to_char = {}

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


def text_to_vec(text, char_to_id):
    """
    Converts text to vector in corpus.
    When char in text is not in corpus, then raises KeyError.

    :param text: text to vector
    :param char_to_id: dictionary of character to id
    :return: vector (dim 1 of numpy array)
    >>> text = "IKEA"
    >>> char_to_id = {'I': 0, 'K': 1, 'E': 2, 'A': 3, '港': 4, '北': 5, '\\t': 6}
    >>> text_to_vec(text, char_to_id)
    array([0, 1, 2, 3])

    >>> text = "IKE!" # "!" is not in char_to_id.
    >>> char_to_id = {'I': 0, 'K': 1, 'E': 2, 'A': 3, '港': 4, '北': 5, '\\t': 6}
    >>> text_to_vec(text, char_to_id)
    Traceback (most recent call last):
        ...
    KeyError: '!'
    """
    chars = [char for char in list(text)]
    return np.array([char_to_id[char] for char in chars])


def convert_one_hot(corpus, vocabulary_size):
    """
    Converts to one-hot vectors.

    :param corpus: list of word ids (dim 1 of numpy array)
    :param vocabulary_size: vocabulary size
    :return: one-hot vectors (dim 2 of numpy array)
    >>> corpus = np.array([1, 2, 1])
    >>> vocabulary_size = 5
    >>> convert_one_hot(corpus, vocabulary_size)
    array([[0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0]], dtype=int32)

    >>> corpus = np.array([[1, 2, 1], [1, 2, 3]])
    >>> vocabulary_size = 5
    >>> convert_one_hot(corpus, vocabulary_size)
    Traceback (most recent call last):
        ...
    ValueError: Corpus must be dim 1, but actual is 2

    >>> corpus = np.array([1, 2, 6])
    >>> vocabulary_size = 5
    >>> convert_one_hot(corpus, vocabulary_size)
    Traceback (most recent call last):
        ...
    IndexError: index 6 is out of bounds for axis 1 with size 5
    """
    N = corpus.shape[0]

    if corpus.ndim != 1:
        raise ValueError(f"Corpus must be dim 1, but actual is {corpus.ndim}")

    one_hot = np.zeros((N, vocabulary_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        one_hot[idx, word_id] = 1

    return one_hot


def generate_npy(char_to_id, output_file):
    """
    >>> texts = ["IKEA港北", "IKEA三郷", "IKEA立川"]
    >>> _, char_to_id, _ = preprocess(texts)
    >>> output_file = "/tmp/generate_npy.npy"
    >>> generate_npy(texts, char_to_id, output_file)
    array([[0, 1, 2, 3, 4, 5],
           [0, 1, 2, 3, 6, 7],
           [0, 1, 2, 3, 8, 9]])
    """
    # TODO: text の長さが違っていてもできるようにする
    # もしかしたら、np.arrayのpython.listでもよいのかもしれない
    # corpus : [array([0, 1, 2, 3, 4, 5]), array([0, 1, 2, 3, 6, 7]), array([0, 1, 2, 3, 8, 9])]
    texts_vec = [np.array([char_to_id[char] for char in text]).reshape(len(text)) for text in texts]
    corpus = np.array(texts_vec)
    np.save(output_file, corpus)
    return corpus