import numpy as np

def normalization( x):
    _min = np.min(x, axis=0)
    _range = np.max(x, axis=0) - _min
    x_norm = (x - _min) / _range
    return _min

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(normalization(x))