import numpy as np

class Perceptron:
    def __init__(
        self,
        n_feature=1,
        epoches=200,
        lr=0.01,
        tol=0.01,
        wandb=False,
        minibatch_size = 100
    ):
        self.epoches = epoches
        self.lr = lr
        self.tol = tol
        self.W = (np.random.random(n_feature + 1) * 0.5).reshape(-1, 1)
        self.best_loss = np.inf
        self.patience = 100
        self.wandb = wandb
        self.minibatch_size = minibatch_size
        self.losses = []

    def _preprocess_data(self, input):
        row, col = input.shape
        input_ = np.empty([row, col + 1])
        input_[:, 0] = 1
        input_[:, 1:] = input
        return input_

    def _feed_forward(self, input):
        y = np.dot(input, self.W)
        return np.sign(y)

    def _loss(self, y_pred, groundtruth):
        loss = y_pred * groundtruth
        loss_all = -loss[loss < 0]
        return np.sum(loss_all)

    def _gradient(self, inputs, outputs, groundtruths):
        batch_size = inputs.shape[0]
        grads = np.zeros_like(inputs)

        for i in range(batch_size):
            input = inputs[i]
            output = outputs[i]
            groundtruth = groundtruths[i]
            grad = (
                -groundtruth * input.reshape(-1, 1)
                if output * groundtruth < 0
                else np.zeros_like(input)
            )
            # print(grad.shape)
            grads[i] = grad.reshape(-1)

        # 求平均梯度
        avg_grad = np.mean(grads, axis=0).reshape(-1, 1)
        return avg_grad


    def fit(self, input, groundtruth):
        input = self._preprocess_data(input)
        num_samples = input.shape[0]
        for epoch in range(self.epoches):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            start_idx = np.random.randint(0, num_samples - self.minibatch_size)

            y_pred = self._feed_forward(input)

            grad = self._gradient(input[start_idx:start_idx+self.minibatch_size], y_pred[start_idx:start_idx+self.minibatch_size], groundtruth[start_idx:start_idx+self.minibatch_size])
            self.W = self.W - self.lr * grad



            loss = self._loss(y_pred, groundtruth)
            self.losses.append(loss)
            if epoch % 50 == 0:
                print(f"epoch: {epoch}, loss: {loss}")

            

    # get accuracy recall precision and F1 score
    def evaluate(self, input, groundtruth):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        input = self._preprocess_data(input)
        for i in range(len(input)):
            y = self._feed_forward(input[i])
            if y == 1 and groundtruth[i] == 1:
                TP += 1
            elif y == 1 and groundtruth[i] == -1:
                FP += 1
            elif y == -1 and groundtruth[i] == 1:
                FN += 1
            elif y == -1 and groundtruth[i] == -1:
                TN += 1
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * precision * recall / (precision + recall)

        print(
            f"evaluation results: accuracy: {accuracy}, recall: {recall}, precision: {precision}, F1: {F1}"
        )

    

    