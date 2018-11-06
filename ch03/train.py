import sys
sys.path.append('..')
from common.util import preprocess, create_contexts_target


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)


def check_corpus():
    """
    >>> check_corpus()
    [0 1 2 3 4 1 5 6]
    """
    print(corpus)


def check_id_to_word():
    """
    >>> check_id_to_word()
    {0: 'You', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'I', 5: 'hello', 6: '.'}
    """
    print(id_to_word)


contexts, target = create_contexts_target(corpus, window_size=1)

def check_contexts():
    """
    >>> check_contexts()
    [[0 2]
     [1 3]
     [2 4]
     [3 1]
     [4 5]
     [1 6]]
    """
    print(contexts)


def check_target():
    """
    >>> check_target()
    [1 2 3 4 1 5]
    """
    print(target)
