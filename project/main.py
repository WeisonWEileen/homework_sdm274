# copyright Weison Pan 2024
import sys
import hydra
from omegaconf import DictConfig
from utils.utils import *
import matplotlib.pyplot as plt

sys.path.append("../")

from engine.lr import bc_LinearRegression
from engine.logRe import LogReg
from engine.petr import Perceptron
from engine.mlp import MLP


@hydra.main(version_base="1.3", config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    
    
    X_train, X_test, y_train, y_test = data_preprocessing("./data/ai4i2020.csv")
    
    # =======================Linear Regression=============
    model_lr = bc_LinearRegression(n_feature=X_train.shape[1], 
    epoches=5000,
    gd_strategy="MBGD", 
    prepro_strategy="min_max",
    learning_rate = 0.1)
    # X_train = model_lr.normalization(X_train)
    # X_test = model_lr.normalization(X_test)
    # plot_loss_curve(model_lr.loss, "Linear Regression")
    # X_train = model_lr.normalization(X_train)
    # model_lr.fit(X_train, y_train)
    # model_lr.evaluate(X_test, y_test, debo=0.1)
    # ====================Linear Regression=========

    ## =================LogReg================
    # model_log = LogReg(
    #     n_feature=X_train.shape[1],
    #     epoches=6000,
    #     lr=0.01,
    #     tol=1e-5,
    #     wandb=False,
    #     gd_strategy="MiniBGD",
    #     mini_batchsize=10,
    # )

    # model_log.train(min_max_normalization(X_train), y_train)
    # model_log.evaluate(min_max_normalization(X_test), y_test, 0.08)
    # plot_loss_curve(model_log.losses, "Logestic Regression")
    ## =================LogReg==================

    ## =================Perceptron=================
    # model_petr = Perceptron(
    #     n_feature=X_train.shape[1], 
    #     epoches=8000, 
    #     lr=0.01, 
    #     tol=0.001, 
    #     wandb=False, 
    #     minibatch_size=100)
    # y_train[y_train == 0] = -1
    # y_test[y_test == 0] = -1
    # model_petr.fit(min_max_normalization(X_train), y_train)
    # model_petr.evaluate(min_max_normalization(X_test), y_test)
    # plot_loss_curve(model_petr.losses, "Perceptron")
    ## =================Perceptron==================

    ## ==============MLP==================
    model_mlp = MLP(units=[3,10,1])
    model_mlp.train(
        min_max_normalization(X_train),
        y_train,
        epoches=3000,
        lr=0.01,
        mini_batchsize=50,
    )
    plot_loss_curve(model_mlp.losses, "MLP")
    model_mlp.evaluate_pr(min_max_normalization(X_test), y_test, np.arange(0.05, 0.25, 0.01))

    # model_mlp.evaluate(min_max_normalization(X_test), y_test)
    ## ==============MLP=================


if __name__ == "__main__":
    main()
