# copyright Weison Pan 2024
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig
from utils.utils import *
from engine.model import MLP
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def complex_nonlinear_function(x):
    return 10 *np.sin(2 * 3.14 * x) + 0.05 * x**3 + np.random.randn(x.shape[0]) * 0.1

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    if(cfg.wandb_on_off):
        wandb.init(project="hw5", config=cfg)



    range = [0, 1]
    points = 100
    x_train = np.linspace(range[0], range[1], points)
    y_train = complex_nonlinear_function(x_train) + np.random.randn(points) * 0.1
    x_eval = np.linspace(0, 1, 50).reshape(-1, 1)
    y_eval = complex_nonlinear_function(x_eval)

    kf = KFold(n_splits=5)
    best_mse = float("inf")
    best_params = None

    for units in cfg.units_list:
        for lr in cfg.lr_list:
            mse_list = []
            for train_index, val_index in kf.split(x_train):
                x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                model = MLP(units=units)
                model.train(
                    x_train_fold.reshape(-1, 1),
                    y_train_fold.reshape(-1, 1),
                    epoches=cfg.epoches,
                    lr=lr,
                    mini_batchsize=cfg.mini_batchsize,
                )

                y_val_pred = model.forward(x_val_fold.reshape(-1, 1))
                mse = mean_squared_error(y_val_fold, y_val_pred)
                mse_list.append(mse)

            avg_mse = np.mean(mse_list)
            print(f"Units: {units}, LR: {lr}, Avg MSE: {avg_mse}")

            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = {"units": units, "lr": lr}

    print(f"Best Params: {best_params}, Best MSE: {best_mse}")

    # 使用最佳参数训练最终模型
    model = MLP(n_units=best_params["units"])
    model.train(
        x_train.reshape(-1, 1),
        y_train.reshape(-1, 1),
        epoches=cfg.epoches,
        lr=best_params["lr"],
        mini_batchsize=cfg.mini_batchsize,
    )

    visualize_fitting_line(x_train, y_train, x_eval, y_eval, model)

    # model.evaluate(x_test, y_test)

if __name__ == "__main__":
    main()
