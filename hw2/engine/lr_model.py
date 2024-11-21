import numpy as np
import wandb

class LinearRegresssion:
    def __init__(
        self,
        n_feature=1,
        learning_rate=0.001,
        batch_size=10,
        epoches=10000,
        tolerance=1e-5,
        gd_strategy="MBGD",
        prepro_strategy="None",
        wandb=False,
    ):
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.tole = tolerance
        self.batch_size = batch_size
        self.W = (np.random.randn(n_feature + 1) * 0.05).reshape(-1, 1)
        self.gd_strategy = gd_strategy
        self.prepro_strategy = prepro_strategy
        self.wandb = wandb

        if self.gd_strategy == "SGD":
            self.update = self._SGD_update
        elif self.gd_strategy == "BGD":
            self.update = self._BGD_update
        elif self.gd_strategy == "MBGD":
            self.update = self._MBGD_update
        else:
            raise ValueError(f"Unknown gradient descent strategy: {self.gd_strategy}")

        if self.prepro_strategy == "min_max":
            self.preprocess = self.normalization
        elif self.prepro_strategy == "mean":
            self.preprocess = self._standardization
        elif self.prepro_strategy == "none":
            self.preprocess = self._extend
        else:
            raise ValueError(f"Unknown preprocessing strategy: {self.prepro_strategy}")

    # stochastic gradient descent
    def _SGD_update(self, input, groundtruth):
        pred = input.dot(self.W)
        seed = np.random.randint(0, len(input))
        grad = self._gradient(
            groundtruth[seed], pred[seed], input[seed, :][np.newaxis, :]
        )  # to ensure the shape is (,2) instead of (2,)
        if self.wandb:
            wandb.log({"grad_0": grad[0],"grad_1": grad[1]})
        
        self.W = self.W - self.learning_rate * grad

    # batch gradient descent
    def _BGD_update(self, input, groundtruth):
        pred = input.dot(self.W)
        grad = self._gradient(groundtruth, pred, input)
        self.W = self.W - self.learning_rate * grad       
        if self.wandb:
            wandb.log({"grad_0": grad[0],"grad_1": grad[1]})

    # 10 percent of mini batch graident descent
    def _MBGD_update(self, input, groundtruth):
        pred = input.dot(self.W)
        indices = np.random.choice(groundtruth.shape[0], self.batch_size, replace=False)
        grad = self._gradient(groundtruth[indices], pred[indices], input[indices])
        if self.wandb:
            wandb.log({"grad_0": grad[0]})
            wandb.log({"grad_1": grad[1]})
        self.W = self.W - self.learning_rate * grad

    def _mse_loss(self, groundtruth, predict):
        return np.mean((groundtruth - predict) ** 2)

    def _gradient(self, groundtruth, predict, input):
        grad = 1 / input.shape[0] * np.dot(input.T, (predict - groundtruth))
        grad = grad.reshape(self.W.shape)
        return grad

    def normalization(self, x):
        _min = np.min(x, axis=0)
        _range = np.max(x, axis=0) - _min
        x_norm = (x - _min) / _range
        return self._extend(x_norm)

    def _standardization(self, x):
        _mu = np.mean(x, axis=0)
        _sigma = np.std(x, axis=0)
        x_norm = (x - _mu) / _sigma
        return self._extend(x_norm)

    def _extend(self, X):
        return np.c_[X, np.ones(len(X))]

    def fit(self, X, y):
        for i in range(self.epoches):
            self.update(X, y)
            if i % 100 == 0:
                result = X.dot(self.W) 
                loss = self._mse_loss(y, result)
                if self.wandb:
                    wandb.log({"loss": loss})
                print(f"for {i} iteration, the loss is {loss}")
        return self.W
