import numpy as np

a = np.array([0, 1, 2])
b = np.array(a)
c = np.array([[10,3,3], [3,4,5], [6,7,8]])

def softmax(x):
    exp_x = np.exp(x- np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_raw(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_wrong(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
# a = a**2
# c = np.sum(c)
# print(a.shape)

print(softmax(c))
print(softmax_raw(c))
print(softmax_wrong(c))