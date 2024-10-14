import numpy as np
import matplotlib.pyplot as plt
import wandb


class Perceptron:
    def __init__(self, n_feature=1, epoches=200, lr=0.01, tol=0.01, wandb=False, gd_strategy="SGD"):
        self.epoches = epoches
        self.lr = lr
        self.tol = tol
        self.W = (np.random.random(n_feature + 1) * 0.5).reshape(-1, 1)
        self.best_loss = np.inf
        self.patience = 100
        self.wandb = wandb
        self.gd_strategy = gd_strategy
        

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
        return  np.sum(loss_all)

    def _gradient(self, inputs, outputs, groundtruths):
        batch_size = inputs.shape[0]
        grads = np.zeros_like(inputs)
    
        for i in range(batch_size):
            input = inputs[i]
            output = outputs[i]
            groundtruth = groundtruths[i]
            grad = - groundtruth * input.reshape(-1, 1) if output * groundtruth < 0 else np.zeros_like(input)
            grads[i] = grad.reshape(-1)
    
        # 求平均梯度
        avg_grad = np.mean(grads, axis=0).reshape(-1, 1)
        return avg_grad
        
    def SGD_fit(self, input, groundtruth):
        ep_no_impro_count = 0
        input = self._preprocess_data(input)
        for epoch in range(self.epoches):
            y_pred = self._feed_forward(input)
            loss = self._loss(y_pred, groundtruth)
            if self.wandb:
                wandb.log({"loss": loss})
            if (loss - self.best_loss) < - self.tol:
                self.best_loss = loss
                ep_no_impro_count = 0
            elif np.abs(loss - self.best_loss) < self.tol:
                ep_no_impro_count += 1
                if ep_no_impro_count >= self.patience:
                    print(f"early stopping at epoch {epoch}")
                    break
            else:
                ep_no_impro_count = 0

            if self.gd_strategy == "SGD":
                i = np.random.randint(0, len(input))
                grad = self._gradient(np.expand_dims(input[i], axis=0), np.expand_dims(y_pred[i], axis=0), np.expand_dims(groundtruth[i], axis=0))
            else:
                grad = self._gradient(input, y_pred, groundtruth)
            self.W = self.W - self.lr * grad
            

    def _BGD_update(self, input, groundtruth):
        grad = self._gradient(input, groundtruth)
        self.W = self.W - self.lr * grad
        if self.wandb:
            wandb.log({"grad_0": grad[0], "grad_1": grad[1]})
        self.W = self.W - self.lr * grad


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

        if self.wandb:
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
