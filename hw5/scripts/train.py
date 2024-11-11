# copyright Weison Pan 2024
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig
import utils.utils as uts
from engine.model import MLP

def complex_nonlinear_function(x):
    return np.sin(x) + 0.05 * x**3

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    if(cfg.wandb_on_off):
        wandb.init(project="hw3", config=cfg)

    x_train = np.linspace(-10, 10, 2000)
    y_train = complex_nonlinear_function(x_train) + np.random.randn(2000) * 0.1

    x_eval = np.linspace(-10, 10, 100)
    y_eval = complex_nonlinear_function(x_eval) 

    model = MLP(
        n_units=[1, 3, 4, 8, 3, 1],
    )

    model.train(
        x_train, y_train, epoches=cfg.epoches, lr=cfg.lr, gd_strategy=cfg.gd_strategy, mini_batchsize=cfg.mini_batchsize)
    # model.evaluate(x_test, y_test)

if __name__ == "__main__":
    main()
