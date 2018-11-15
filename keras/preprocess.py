# -*- coding: utf-8 -*-
import re
import numpy as np


def preprocess(texts):
    """
    Preprocessor for text to char vector
    :param texts: raw corpus
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
    """
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
    ignore_chars = [' ', '']
    # TODO: normalize Japanese.
    return [char for char in list(text) if char not in ignore_chars]


def generate_npy(texts, char_to_id, output_file):
    """
    >>> texts = ["IKEA港北", "IKEA新三郷", "IKEA立川"]
    >>> _, char_to_id, _ = preprocess(texts)
    >>> output_file = "/tmp/generate_npy.npy"
    >>> generate_npy(texts, char_to_id, output_file)
    (3, 1)
    """
    texts_vec = [np.array([char_to_id[char] for char in text]) for text in texts]
    corpus = np.array(texts_vec).reshape(len(texts_vec), 1)
    np.save(output_file, corpus)
    print(corpus.shape)

