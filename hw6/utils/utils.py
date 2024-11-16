import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''
read Wisconsin Diagnostic Breast Cancer dataset
'''
def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, 2:].values  # 取第3列到最后一列作为特征
    y = data.iloc[:, 1].values  # 取第2列作为标签
    y = np.where(y == 'M', 1, 0)
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def evaluate(y, groundtruth):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # input = self._preprocess_data(input)
    for i in range(len(y)):
        if y[i] >= 0.5 and groundtruth[i] == 1:
            TP += 1
        elif y[i] >= 0.5 and groundtruth[i] == 0:
            FP += 1
        elif y[i] < 0.5 and groundtruth[i] == 1:
            FN += 1
        elif y[i] < 0.5 and groundtruth[i] == 0:
            TN += 1
        else:
            print(f"error: y: {y[i]}, groundtruth: {groundtruth[i]}")
    print(f"all cases: {y.shape[0]}, TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = 2 * precision * recall / (precision + recall)

    print(
        f"evaluation results: accuracy: {accuracy}, recall: {recall}, precision: {precision}, F1: {F1}"
    )

    return accuracy, recall, precision, F1


if __name__ == '__main__':
    file_path = 'data/wdbc.data'
    read_data(file_path)