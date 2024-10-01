import numpy as np

class LinearRegresssion:
    def __init__(self, n_feature=1, learning_rate=0.0001, batch_size=10, epoches=10000,tolerance=1e-5, gd_strategy='MBGD'):
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.tole = tolerance
        self.batch_size = batch_size
        self.W = np.random.randn(n_feature + 1) * 0.05
        self.gd_strategy = gd_strategy
        
        if self.gd_strategy == 'SGD':
            self.update = self._SGD_update
        elif self.gd_strategy == 'BGD':
            self.update = self._BGD_update
        elif self.gd_strategy == 'MBGD':
            self.update = self._MBGD_update
        else:
            raise ValueError(f"Unknown gradient descent strategy: {self.gd_strategy}")

    # stochastic gradient descent
    def _SGD_update(self, input, groundtruth):
        pred = input.dot(self.W)
        seed = np.random.randint(0, len(input))
        grad = self._gradient(
            groundtruth[seed], pred[seed], input[seed, :][np.newaxis, :]
        )  # to ensure the shape is (,2) instead of (2,)
        self.W = self.W - self.learning_rate * grad

    # batch gradient descent
    def _BGD_update(self, input, groundtruth):
        pred = input.dot(self.W)
        grad = self._gradient(groundtruth, pred, input)
        self.W = self.W - self.learning_rate * grad

    # 10 percent of mini batch graident descent
    def _MBGD_update(self, input, groundtruth):
        pred = input.dot(self.W)
        indices = np.random.choice(groundtruth.shape[0], self.batch_size, replace=False)
        grad = self._gradient(groundtruth[indices], pred[indices], input[indices])
        self.W = self.W - self.learning_rate * grad

    def _mse_loss(self, groundtruth, predict):
        return np.mean((groundtruth - predict) ** 2)

    def _gradient(self, groudtruth, predict, input):
        return 1 / input.shape[0] * np.mean(2 * (predict - groudtruth) * input[:, :-1])

    def fit(self, X, y):
        X = np.c_[X, np.ones(len(X))]
        for i in range(self.epoches):
            self.update(X, y)
            if i % 100 == 0:
                loss = self._mse_loss(y, X.dot(self.W))
                print(f"for {i} iteration, the loss is {loss}")
        return self.W
