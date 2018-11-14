# coding: utf-8
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


# set hyper parameters
batch_size = 10
wordvec_size = 100
# element size of hidden vector for RNN
hidden_size = 100
# size to expand RNN
time_size = 5
lr = 0.1
max_epoch = 100

# load corpus
corpus, word_to_id, id_to_word = ptb.load_data('train')
# shrink test data set
corpus_size = 1000
corpus = corpus[:corpus_size]
vocabulary_size = int(max(corpus) + 1)
# input
xs = corpus[:-1]
# output (trainer label)
ts = corpus[1:]

# generate model
model = SimpleRnnlm(vocabulary_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# run
trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()