# copyright Weison Pan 2024
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from engine.lr_model import LinearRegresssion



@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # generate some data
    X_train = np.arange(100).reshape(100,1)
    a, b = 1, 10
    y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)
    y_train = y_train.reshape(-1,1)

    lr = cfg.learning_rate
    bs = cfg.batch_size
    ep = cfg.epoches
    tol = cfg.tolerance
    gd_s = cfg.gd_strategy
    pre_s = cfg.prepro_strategy
    wandb_on_off = cfg.wandb_on_off

    if wandb_on_off:
        wandb.init(project=cfg.wandb.project,dir=cfg.wandb.dir,config=cfg)
    model = LinearRegresssion(n_feature=X_train.shape[1], learning_rate=lr, batch_size=bs, epoches=ep,tolerance=tol,gd_strategy=gd_s,prepro_strategy=pre_s,wandb=wandb_on_off)
    
    X_train = model.preprocess(X_train)
    plt.scatter(X_train[:, 0], y_train, color='blue', label='Training data')
    w = model.fit(X_train, y_train)
    if wandb_on_off:
        wandb.finish()

    x_values = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    y_values = w[0] * x_values + w[1]

    plt.plot(x_values, y_values, color='red', label='Regression line')
    plt.xlabel('X_train')
    plt.ylabel('y_train')
    plt.legend()
    plt.title('Linear Regression Fit')
    plt.show()


if __name__ == "__main__":
    main()
