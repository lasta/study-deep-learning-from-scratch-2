# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from common.np import np


def sigmoid(x):
    """
    Return the sigmoid of x.
    One of the activation functions.

    >>> sigmoid(0)
    0.5
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    Return more than or equal 0.
    When x is negative, it returns 0.

    >>> x = 1
    >>> relu(x)
    1
    >>> x = 0
    >>> relu(x)
    0
    >>> x = -1
    >>> relu(x)
    0
    """
    return np.maximum(0, x)


def softmax(x):
    """
    Softmax function.
    Score to Probability.
    Returns float [0.0, 1.0]

    >>> x = np.array([0])
    >>> softmax(x)
    array([1.])
    >>> x = np.array([0, 0])
    >>> softmax(x)
    array([0.5, 0.5])
    >>> x = np.array([[0, 0], [0, 0]])
    >>> softmax(x)
    array([[0.5, 0.5],
           [0.5, 0.5]])
    """

    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x = x / x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    """
    Cross entropy error.
    One of the loss functions.
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # convert to index of correct label when training data is one-hot-vector
    # (教師データが one-hot-vector の場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return - np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size



