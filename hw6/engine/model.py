import numpy as np
from scipy.spatial import KDTree
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.kdtree = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.kdtree = KDTree(X)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        dist, idx = self.kdtree.query(x, k=self.k, p=2)
        if idx.ndim == 1:
            idx = [idx]
        elif idx.ndim == 0:
            idx = [np.array([idx])]
        neighbors_labels = [self.y_train[i] for i in idx[0]]
        prediction = max(set(neighbors_labels), key=neighbors_labels.count)
        return prediction
