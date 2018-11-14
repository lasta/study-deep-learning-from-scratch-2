# coding: utf-8
import sys
sys.path.append('..')
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


# hyper parameters
batch_size = 10
wordvec_size = 100
hidden_size = 100
# time size for Truncated BPTT expansion
time_size = 5
lr = 0.1
max_epoch = 100


# load training data (reduce dataset size)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocabulary_size = int(max(corpus) + 1)

# input
xs = corpus[:-1]
# output (training label)
ts = corpus[1:]
data_size = len(xs)
print(f"corpus size: {corpus_size}, vocabulary size: {vocabulary_size}.")

# variables for training
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_idx = 0
loss_count = 0
ppl_list = []

# generate model
model = SimpleRnnlm(vocabulary_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# calc position of start to read each mini-batch samples.
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # get mini-batch
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # calculate gradient and update parameters
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # evaluate perplexity each epoch
    ppl = np.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f' % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0


# グラフの描画
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()
