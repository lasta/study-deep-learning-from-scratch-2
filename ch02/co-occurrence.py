#!/usr/bin/env python

import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi
import numpy as np


text = 'You say goodbye and I say hello.'.lower()
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
print(id_to_word)


vocabulary_size = len(word_to_id)
C = create_co_matrix(corpus=corpus, vocabulary_size=vocabulary_size)

print(vocabulary_size)
print(C)

vec_you = C[word_to_id['you']]
vec_i   = C[word_to_id['i']]
vec_hello = C[word_to_id['hello']]
vec_say = C[word_to_id['say']]
vec_goodbye = C[word_to_id['goodbye']]
vec_and = C[word_to_id['and']]

print('you, i')
print(cos_similarity(vec_you, vec_i))
print('you, hello')
print(cos_similarity(vec_you, vec_hello))
print('you, say')
print(cos_similarity(vec_you, vec_say))
print('you, goodbye')
print(cos_similarity(vec_you, vec_goodbye))
print('you, and')
print(cos_similarity(vec_you, vec_and))


most_similar('you', word_to_id, id_to_word, C, top=5)

W = ppmi(C, verbose=True)
np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-' * 50)
print('PPMI')
print(W)
