# copyright Weison Pan 2024
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig
from engine.model import KNN
from utils.utils import *
import matplotlib.pyplot as plt


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    X, y = read_data(cfg.dataset_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(" from 1 to 10")

    Accuracy = [] 
    Recall = [] 
    Precision = [] 
    F1s = []
    for i in range(10):
        print(f"K = {i+1}")
        knn = KNN(i+1)

        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        accuracy, recall, precision, F1 = evaluate(y_pred, y_test)
        Accuracy.append(accuracy)
        Recall.append(recall)
        Precision.append(precision)
        F1s.append(F1)
    
    k_values = list(range(1, 11))

    plt.figure(figsize=(12, 8))
    plt.plot(k_values, Accuracy, label='Accuracy', marker='o')
    plt.plot(k_values, Recall, label='Recall', marker='o')
    plt.plot(k_values, Precision, label='Precision', marker='o')
    plt.plot(k_values, F1s, label='F1 Score', marker='o')
    plt.title('Evaluation Metrics for k-NN with Different k Values')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
