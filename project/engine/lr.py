from utils.utils import *
from hw2.engine.lr_model import LinearRegresssion

# inherit from LinearRegresssion from hw2
class bc_LinearRegression(LinearRegresssion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = []
    
    def fit(self, X, y):
        for i in range(self.epoches):
            self.update(X, y)
            if i % 100 == 0:
                result = X.dot(self.W) 
                loss = self._mse_loss(y, result)
                print(f"for {i} iteration, the loss is {loss}")
                self.loss.append(loss)
        return self.W
    
    def evaluate(self, x, groundtruth, debo=0.5):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        # X = self.preprocess(x)
        y_pred = x.dot(self.W) 
        for i in range(len(y_pred)):
            print()
            if y_pred[i] >= debo and groundtruth[i] == 1:
                TP += 1
            elif y_pred[i] >= debo and groundtruth[i] == 0:
                FP += 1
            elif y_pred[i] < debo and groundtruth[i] == 1:
                FN += 1
            elif y_pred[i] < debo and groundtruth[i] == 0:
                TN += 1
            else:
                print(f"error: y: {y_pred[0][0]}, groundtruth: {groundtruth[i][0]}")
        
        print(f"total cases num(y=0): {len(groundtruth[groundtruth==0])}, num(y=1): {len(groundtruth[groundtruth==1])}")


        print(f"all cases: {x.shape[0]}, TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        # exit()
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * precision * recall / (precision + recall)  
        print(
            f"evaluation results: accuracy: {accuracy}, recall: {recall}, precision: {precision}, F1: {F1}"
        )


    
