# copyright Weison Pan 2024
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig
from utils.utils import *
from engine.model import MLP
import matplotlib.pyplot as plt


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    if(cfg.wandb_on_off):
        wandb.init(project="hw5", config=cfg)

    range = [0,1]
    points = 100
    x_train = np.linspace(range[0], range[1], points)
    y_train = complex_nonlinear_function(x_train) + np.random.randn(points) * 0.1
    x_eval = np.linspace(0, 1, 50).reshape(-1,1)
    y_eval = complex_nonlinear_function(x_eval) 

    model = MLP(units=cfg.units)
    if cfg.gd_strategy == "MiniBGD":
        mini_batchsize = cfg.mini_batchsize
    elif cfg.gd_strategy == "SGD":
        mini_batchsize = 1
    else:
        raise ValueError("Invalid gd_strategy")

    model.train(
        x_train.reshape(-1, 1),
        y_train.reshape(-1, 1),
        epoches=cfg.epoches,
        lr=cfg.lr,
        mini_batchsize=mini_batchsize,
    )

    visualize_fitting_line(x_train, y_train, x_eval, y_eval, model)

    # model.evaluate(x_test, y_test)

if __name__ == "__main__":
    main()
