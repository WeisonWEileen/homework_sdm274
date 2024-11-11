import numpy as np
import wandb
import matplotlib.pyplot as plt

"""
numpy实现的 mlp 类。input: number of layers 和 units per layer (一个 list)
"""


class MLP:
    def __init__(self, n_units=[1, 1]):
        self.best_loss = np.inf
        self.n_layers = len(n_units)
        self.n_units = n_units
        # to store temporary values to calculate gradients
        self.no_grad = False
        self.a = []
        self.deltas = []
        self.weights = []
        self.biases = []
        self.dz = 0
        for i in range(self.n_layers-1):
            self.weights.append(np.random.randn(n_units[i], n_units[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, n_units[i + 1])))

    def _linear_tf(self, W, X, b):
        return np.dot(X, W) + b

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _d_sigmoid(self, x):
        return x * (1 - x)

    def _forward(self, X):
        self.a = []
        self.a.append(X)
        for i in range(len(self.weights) -1):
            z = self._linear_tf(self.weights[i], self.a[-1], self.biases[i])
            a = self._sigmoid(z)
            self.a.append(a)
        output = self._linear_tf(self.weights[-1], self.a[-1], self.biases[-1])
        self.a.append(output)
        return output

    def _mse_loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    # deltas 是指上一层对输出求偏导的结果
    '''
    self.deltas : 储存对每一层求偏导的结果
    '''
    def _backward(self, y, lr):
        batch_size = y.shape[0]
        self.deltas = []
        self.deltas.append(self.a[-1] - y)
        for i in range(len(self.a) - 2, 0, -1):
            delta = np.dot(self.deltas[-1], self.weights[i].T) * self._d_sigmoid(self.a[i])
            self.deltas.append(delta)
        self.deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= lr * np.dot(self.a[i].T, self.deltas[i]) / batch_size
            self.biases[i] -= lr * np.sum(self.deltas[i], axis=0, keepdims=True) / batch_size

    def train(self, input, groundtruth, epoches=1000, lr=0.01, gd_strategy="MiniBGD", mini_batchsize=10):
        length = input.shape[0]
        losses = []

        for epoch in range(epoches):
            indices = np.arange(input.shape[0])
            np.random.shuffle(indices)
            for start in range(0, input.shape[0], mini_batchsize):
                end = start + mini_batchsize
                X_batch, y_batch = input[indices[start:end]], groundtruth[indices[start:end]]
                self._forward(X_batch)
                self._backward(X_batch, y_batch, lr)
            loss = self._mse_loss(self._forward(input), groundtruth)
            losses.append(loss)
            if epoch % 100 == 0:            
                print(f'Epoch {epoch}, Loss: {loss}')

        # 绘制损失函数变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

    # get accuracy recall precision and F1 score
    def evaluate(self, input, groundtruth):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        input = self._preprocess_data(input)
        for i in range(len(input)):
            y = self._feed_forward(input[i])
            if y >= 0.5 and groundtruth[i] == 1:
                TP += 1
            elif y >= 0.5 and groundtruth[i] == 0:
                FP += 1
            elif y < 0.5 and groundtruth[i] == 1:
                FN += 1
            elif y < 0.5 and groundtruth[i] == 0:
                TN += 1
            else:
                print(f"error: y: {y[0][0]}, groundtruth: {groundtruth[i][0]}")
        print(f"all cases: {input.shape[0]}, TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

        accuracy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * precision * recall / (precision + recall)

        if wandb:
            wandb.log(
                {
                    "accuracy": accuracy,
                    "recall": recall,
                    "precision": precision,
                    "F1": F1,
                }
            )
        print(
            f"evaluation results: accuracy: {accuracy}, recall: {recall}, precision: {precision}, F1: {F1}"
        )
