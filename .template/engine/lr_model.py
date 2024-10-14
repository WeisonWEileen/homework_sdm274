import numpy as np
import wandb

class LinearRegresssion:
    def __init__(
        self,
        n_feature=1,
        learning_rate=0.0001,
        batch_size=10,
        epoches=10000,
        tolerance=1e-5,
        gd_strategy="MBGD",
        prepro_strategy="None",
        wandb=False,
    )