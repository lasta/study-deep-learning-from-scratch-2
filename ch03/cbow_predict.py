# coding: utf-8
"""
Continuous bag-of-words (CBOW)

MatMul -+
 W_in   |
        |
        +-> [+] -> [x] -> MatMul -> Score
        |           ^     W_out
        |   0.5 ----+
MatMul -+
 W_in
"""
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul


# sample context data
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# initialize weights
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# generate layers
# Common weight W_in for input
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# forward transmission (順伝搬)
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)
print(s)
