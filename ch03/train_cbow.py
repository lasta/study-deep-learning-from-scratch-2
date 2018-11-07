import sys
sys.path.append('..')
from common.util import preprocess, create_contexts_target, convert_one_hot
from simple_cbow import SimpleCBOW
from common.optimizer import Adam
from common.trainer import Trainer


window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000


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


vocabulary_size = len(word_to_id)
target_one_hot = convert_one_hot(target, vocabulary_size)


def check_target_one_hot():
    """
    >>> check_target_one_hot()
    [[0 1 0 0 0 0 0]
     [0 0 1 0 0 0 0]
     [0 0 0 1 0 0 0]
     [0 0 0 0 1 0 0]
     [0 1 0 0 0 0 0]
     [0 0 0 0 0 1 0]]
    """
    print(target_one_hot)


contexts_one_hot = convert_one_hot(contexts, vocabulary_size)


def check_contexts_one_hot():
    """
    >>> check_contexts_one_hot()
    [[[1 0 0 0 0 0 0]
      [0 0 1 0 0 0 0]]
    <BLANKLINE>
     [[0 1 0 0 0 0 0]
      [0 0 0 1 0 0 0]]
    <BLANKLINE>
     [[0 0 1 0 0 0 0]
      [0 0 0 0 1 0 0]]
    <BLANKLINE>
     [[0 0 0 1 0 0 0]
      [0 1 0 0 0 0 0]]
    <BLANKLINE>
     [[0 0 0 0 1 0 0]
      [0 0 0 0 0 1 0]]
    <BLANKLINE>
     [[0 1 0 0 0 0 0]
      [0 0 0 0 0 0 1]]]
    """
    print(contexts_one_hot)


model = SimpleCBOW(vocabulary_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer.fit(contexts_one_hot, target_one_hot, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
