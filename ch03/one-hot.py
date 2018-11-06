import numpy as np

# input
c = np.array([[ 1, 0, 0, 0, 0, 0, 0 ]])
print(c)
# weight
W = np.random.randn(7, 3)
print(W)
# intermediate node
h = np.dot(c, W)
print(h)
