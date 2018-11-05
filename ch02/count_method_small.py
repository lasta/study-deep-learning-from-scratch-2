import sys
sys.path.append('..')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocabulary_size = len(id_to_word)
# co-matrix
C = create_co_matrix(corpus, vocabulary_size, window_size=1)
W = ppmi(C)

# SVD; W ~ U * S * Vt
U, S, V = np.linalg.svd(W)

print(corpus)
print(word_to_id)
print(id_to_word)
print('co-occurance matrix')
print(C[0])
print('PPMI matrix')
print(W[0])
print('SVD')
print(U[0])
