# copyright Weison Pan 2024

import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

from engine.lr_model import LinearRegresssion

@hydra.main(version_base="1.3",config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # generate some data
    X_train = np.arange(100).reshape(100,1)
    a, b = 1, 10
    y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)
    y_train = y_train.reshape(-1)

    lr = cfg.learning_rate
    bs = cfg.batch_size
    ep = cfg.epoches
    tol = cfg.tolerance
    gd_s = cfg.gd_strategy
    model = LinearRegresssion(n_feature=X_train.shape[1], learning_rate=lr, batch_size=bs, epoches=ep,tolerance=tol,gd_strategy=gd_s)
    w = model.fit(X_train, y_train)
    plt.scatter(X_train[:, 0], y_train, color='blue', label='Training data')

    x_values = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    y_values = w[0] + w[1] * x_values
    plt.plot(x_values, y_values, color='red', label='Regression line')

    plt.xlabel('X_train')
    plt.ylabel('y_train')
    plt.legend()
    plt.title('Linear Regression Fit')
    plt.show()



if __name__ == "__main__":
    main()
