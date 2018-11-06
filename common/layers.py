# coding: utf-8
from common.np improt * # includes import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backword(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class Affine:
    """
    Affine transformation
    """
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backword(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

        
class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backword(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backword(selft, dout=1):
        batch_size = self_t_shape[0]
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        d *= dout
        dx = dx // batch_size

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backword(self, dout):
        return dout * (1.0 - self.out) * self.out)


class SigmoidWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.loss = None
        # Output by sigmoid
        self.y = None
        # Training data
        self.t = None


    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backword(self, dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) * dout / batch_size


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params = []
        self.grads = []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag=True):
        if not train_flag:
            return x * (1.0 - self.dropout_ratio)

        self.mask = np.random.rand(*x.shape) > self.dropout_ratio
        return x * self.mask

    def backword(self, dout):
        return dout * self.mask


def Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        return W[idx]

    def backword(self, dout):
        dW, = self_grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None