# copyright Weison Pan 2024
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig
import utils.utils as uts
from engine.model import LogReg


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    if(cfg.wandb_on_off):
        wandb.init(project="hw3", config=cfg)

    dataset_path = cfg.dataset_path
    if(uts.count_lines(dataset_path) == 178):
        uts.filter2(dataset_path)
        print("preprocess dataset ")
    y,x = uts.load_data(dataset_path)
    x_train, x_test, y_train, y_test = uts.split_data(x, y, test_size=0.3, random_state=42)

    model = LogReg(
        n_feature=x_train.shape[1],
        epoches=cfg.epoches,
        lr=cfg.lr,
        tol=cfg.tol,
        wandb=cfg.wandb_on_off,
        gd_strategy=cfg.gd_strategy,
        mini_batchsize=cfg.mini_batchsize,
    )
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    main()
