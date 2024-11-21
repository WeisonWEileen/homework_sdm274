import numpy as np
import matplotlib.pyplot as plt

class LogReg:
    def __init__(self, n_feature=1, epoches=200, lr=0.01, tol=0.01, wandb=False, gd_strategy="SGD", mini_batchsize=100):
        self.epoches = epoches
        self.lr = lr
        self.tol = tol
        self.W = (np.random.random(n_feature + 1) * 0.5).reshape(-1, 1)
        self.best_loss = np.inf
        self.patience = 100
        self.wandb = wandb
        self.gd_strategy = gd_strategy
        self.mini_batchsize = mini_batchsize
        self.losses = []

    def _preprocess_data(self, input):
        row, col = input.shape
        input_ = np.empty([row, col + 1])
        input_[:, 0] = 1
        input_[:, 1:] = input
        return input_

    def _linear_tf(self, X):
        return X @ self.W

    def _sigmoid(self,z):
        return 1/(1 + np.exp(-z))

    def _feed_forward(self, X):
        pred = self._sigmoid(self._linear_tf(X))
        return pred.reshape(-1,1)

    def _loss(self, y_pred, gt):
        epsilon = 1e-5
        loss = - np.mean(
            gt * np.log(y_pred + epsilon) + (1-gt) * np.log(1-y_pred + epsilon)
        )
        return loss

    def _gradient(self, X, y_preds, groundtruths):
        grad = - 1 / y_preds.shape[0] * X.T @ (groundtruths - y_preds).reshape(-1, 1)
        # 求平均梯度
        return grad

    def train(self, input, groundtruth):
        input = self._preprocess_data(input)
        for epoch in range(self.epoches):
            y_pred = self._feed_forward(input)
            loss = self._loss(y_pred, groundtruth)
            if epoch % 50 == 0:
                print(f"epoch: {epoch}, loss: {loss}")
            self.losses.append(loss)


            if self.gd_strategy == "SGD":
                i = np.random.randint(0, len(input))
                grad = self._gradient(np.expand_dims(input[i], axis=0), np.expand_dims(y_pred[i], axis=0), np.expand_dims(groundtruth[i], axis=0))
            elif self.gd_strategy == "MiniBGD":
                batch_indices = np.random.choice(
                    len(input), self.mini_batchsize, replace=False
                )
                input_batch = input[batch_indices]
                y_pred_batch = y_pred[batch_indices]
                groundtruth_batch = groundtruth[batch_indices]
                grad = self._gradient(input_batch, y_pred_batch, groundtruth_batch)
                self.W = self.W - self.lr * grad

    # get accuracy recall precision and F1 score
    def evaluate(self, input, groundtruth, threshold=0.5):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        input = self._preprocess_data(input)
        for i in range(len(input)):
            y = self._feed_forward(input[i])
            if y >= threshold and groundtruth[i] == 1:
                TP += 1
            elif y >= threshold and groundtruth[i] == 0:
                FP += 1
            elif y < threshold and groundtruth[i] == 1:
                FN += 1
            elif y < threshold and groundtruth[i] == 0:
                TN += 1
            else:
                print(f"error: y: {y[0][0]}, groundtruth: {groundtruth[i][0]}")
        print(f"total cases num(y=0): {len(groundtruth[groundtruth==0])}, num(y=1): {len(groundtruth[groundtruth==1])}")
        print(f"all cases: {input.shape[0]}, TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * precision * recall / (precision + recall)

        print(
            f"evaluation results: accuracy: {accuracy}, recall: {recall}, precision: {precision}, F1: {F1}"
        )


    def evaluate_pr(self, input, groundtruth, thresholds=np.arange(0.0, 1.1, 0.1)):
        precisions = []
        recalls = []
        input = self._preprocess_data(input)
        for threshold in thresholds:
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(len(input)):
                y = self._feed_forward(input[i])
                if y >= threshold and groundtruth[i] == 1:
                    TP += 1
                elif y >= threshold and groundtruth[i] == 0:
                    FP += 1
                elif y < threshold and groundtruth[i] == 1:
                    FN += 1
                elif y < threshold and groundtruth[i] == 0:
                    TN += 1
                else:
                    print(f"error: y: {y[0][0]}, groundtruth: {groundtruth[i][0]}")
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            
            print(f"Threshold: {threshold}, TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
            print(f"Precision: {precision}, Recall: {recall}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()