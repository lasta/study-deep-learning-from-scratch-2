import sys
sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')

print('corpus size: ', len(corpus))
print('corpus[:30]: ', corpus[:30])
print()
print('id_to_word[0]: ', id_to_word[0])
print('id_to_word[1]: ', id_to_word[1])
print('id_to_word[2]: ', id_to_word[2])
print()
print("word_to_id['car']: ", word_to_id['car'])
print("word_to_id['happy']: ", word_to_id['happy'])
print("word_to_id['lexus']: ", word_to_id['lexus'])


window_size = 2
wordvec_size = 100
vocabulary_size = len(word_to_id)

print('Counting co-occurance ...')
C = create_co_matrix(corpus, vocabulary_size, window_size)

print('Calculating PPMI ...')
W = ppmi(C, verbose=True)

print('Calculating SVD ...')
try:
    print('truncated SVD (fast)')
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

except ImportError:
    print('SVD (slow)')
    U, S, V = np.linalg.svd(W)


word_vecs = U[:, :wordvec_size]

queries = ['you', 'year', 'car', 'toyota']

[ most_similar(query, word_to_id, id_to_word, word_vecs, top=5) for query in queries ]
