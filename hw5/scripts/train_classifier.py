# copyright Weison Pan 2024
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig
from utils.utils import *
from engine.model import MLP
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_gaussian_quantiles
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


X, y = make_blobs(
    n_samples=1000, centers=2, cluster_std=1.0, random_state=42
)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.wandb_on_off:
        wandb.init(project="hw5", config=cfg)


    X,y = make_blobs(
        n_samples=1000, centers=2, cluster_std=5.0, random_state=42
    )
    visualize_blobs(X, y, title="Generated Classification Data")
    y = y.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    kf = KFold(n_splits=5)
    # best_mse = float("inf")
    # best_params = None

    for units in cfg.cla_units_list:
        for lr in cfg.lr_list:
            mse_list = []
            for train_index, val_index in kf.split(x_train):
                x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                model = MLP(units=units)
                model.train(
                    x_train_fold,
                    y_train_fold,
                    epoches=cfg.epoches,
                    lr=lr,
                    mini_batchsize=cfg.mini_batchsize,
                )
                y_val_pred = model.forward(x_val_fold)
                mse = mean_squared_error(y_val_fold, y_val_pred)
                mse_list.append(mse)
            model.evaluate(x_test, y_test)
            avg_mse = np.mean(mse_list)
            print(f"Units: {units}, LR: {lr}, Avg MSE: {avg_mse}")

    # @TODO : plot decision the best boundary, not i plot the last
    # @TODO : change the lost function to cross entropy
    plot_decision_boundary(model, X, y, title="Decision Boundary")


if __name__ == "__main__":
    main()
