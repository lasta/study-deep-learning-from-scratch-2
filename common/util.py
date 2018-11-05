#!/usr/bin/env python

import sys
sys.path.append('..')
import os
import re
from common.np import *


def preprocess(text):
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
    nx = x / np.sqrt(np.sum(x ** 2)) + eps
    ny = y / np.sqrt(np.sum(y ** 2)) + eps
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """
    Calculate most similar words with query
    @param query query
    @param word_to_id dictionary from word to word id
    @param id_to_word dictionary from word id to word
    @param word_matrix matrix that word vectors joined to.
    @param top number how many returns
    @return words that simlar to query 
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

    @param C co-occurrence matrix
    @return PPMI matrix
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


