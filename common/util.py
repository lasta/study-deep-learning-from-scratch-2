#!/usr/bin/env python

import sys
sys.path.append('..')
import os
import re
from common.np import *


def preprocess(text):
    """
    Preprocessor for text to do deep learning.

    >>> text = 'You say goodbye and I say hello.'
    >>> corpus, word_to_id, id_to_word = preprocess(text)
    >>> corpus
    array([0, 1, 2, 3, 4, 1, 5, 6])
    >>> word_to_id
    {'You': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'I': 4, 'hello': 5, '.': 6}
    >>> id_to_word
    {0: 'You', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'I', 5: 'hello', 6: '.'}
    """

    lowered_text = text.lower()
    words = [word for word in re.split('(\W)+?', text) if word not in [' ', '']]

    word_to_id = {}
    id_to_word = {}
    
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[word] for word in words])

    return corpus, word_to_id, id_to_word


def cos_similarity(x, y, eps=1e-8):
    """
    Calculates cosine similarity.
    TODO: add doctest
    """
    nx = x / np.sqrt(np.sum(x ** 2)) + eps
    ny = y / np.sqrt(np.sum(y ** 2)) + eps
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """
    Calculates most similar words with query.

    :param query: query
    :param word_to_id: dictionary from word to word id
    :param id_to_word: dictionary from word id to word
    :param word_matrix: matrix that word vectors joined to.
    :param top: number how many returns
    :return: words that simlar to query 
    """
    # extract query
    if query not in word_to_id:
        print('%s is not found.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # calculate cosine similarity
    vocabulary_size = len(id_to_word)
    similarity = np.zeros(vocabulary_size)
    for i in range(vocabulary_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # sort by the result of cosine similarity
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def convert_one_hot(corpus, vocabulary_size):
    """
    Converts to one-hot vector.

    :param corpus: list of word ids (dim 1 or 2 of numpy array)
    :param vocabulary_size: vocabulary size
    :return: one-hot vector (dim 2 or 3 of numpy array)

    >>> corpus = np.array([1, 2, 1])
    >>> vocabulary_size = 3
    >>> one_hot = convert_one_hot(corpus, vocabulary_size)
    >>> print(one_hot)
    [[0 1 0]
     [0 0 1]
     [0 1 0]]
    """
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocabulary_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocabulary_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def create_co_matrix(corpus, vocabulary_size, window_size=1):
    """
    Creates co-occurrence matrix.
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocabulary_size, vocabulary_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def ppmi(C, verbose=False, eps=1e-8):
    """
    Convert from co-occurrence matrix to PPMI matrix.
    PPMI : Pointwise Mutual Information (ja; 相互情報量)
    PMI(x, y) = log_2 (P(x, y) / ( P(x) P(y) ))

    :param C: co-occurrence matrix
    :return PPMI: matrix
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    count = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                count += 1
                if count % (total // 100) == 0:
                    print('%.1f%% done' % (100 * count / total))

    return M


def create_contexts_target(corpus, window_size=1):
    """
    Transport corpus to one-hot-vectors.

    :param words: NumPy array of word ids
    :param vocabulary_size: size of vocablary
    :return: NumPy array that transported to one-hot-vectors.
    """
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def clip_grads(grads, max_norm):
    """
    Clips gradient.
    :param grads: gradient, mutable
    :param max_norm: max of norm
    """

    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grrad *= rate


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s is not found' % word)
            return
        
    print(f"\n[analogy] {a}:{b} = {c}:?")
    a_vec = word_matrix[word_to_id[a]]
    b_vec = word_matrix[word_to_id[b]]
    c_vec = word_matrix[word_to_id[c]]
    query_vec = normalize(b_vec - a_vec + c_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print(f"==> {answer}: {str(np.dot(word_matrix[word_to_id[answer]], query_vec))}")
    
    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(f" {id_to_word[i]}: {similarity[i]}")

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x