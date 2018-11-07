# -*- coding: utf-8 -*-
"""
Continuous bag-of-words (CBOW)

MatMul -+                 vector ----+
 W_in   |                            v
        |                           Softmax
        +-> [+] -> [x] -> MatMul -> With    -> Loss
        |           ^     W_out     Loss
        |   0.5 ----+
MatMul -+
 W_in
"""
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    """
    Simple continuous bag-of-words.
    """

    def __init__(self, vocabulary_size, hidden_size):
        V, H = vocabulary_size, hidden_size

        # initialize weights
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # generate layers
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # list all weights and gradient layers
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # set distributed representation of words to variable
        self.word_vecs = W_in


    def forward(self, contexts, target):
        """
        :param contexts: dim 3 of numpy array
        :param target: dim2 of numpy array
        """
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        """
        Continuous bag-of-words (CBOW)
                0.5*da
        MatMul <-+                  vector  ----+
         W_in    |                              v
                 |     0.5*da                  Softmax
                 +-- [+] <- [x] <-- MatMul <-- With    <-- Loss
                 |           ^  da  W_out   ds Loss     1
                 |   0.5 ----+
        MatMul <-+
         W_in   0.5*da
        """
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
