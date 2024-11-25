import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def data_preprocessing(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop(columns=['UDI', 'Product ID', 'HDF', 'OSF', 'PWF', 'TWF', 'RNF', 'Rotational speed [rpm]', 'Type', 'Process temperature [K]'])
    y = data['Machine failure'].values.reshape(-1, 1)
    x = data.iloc[:, :3].values
    
    return split_data(x,y)

import matplotlib.pyplot as plt

def plot_loss_curve(loss,  name):
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Loss', marker='o')
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def min_max_normalization(x):
    _min = np.min(x, axis=0)
    _range = np.max(x, axis=0) - _min
    x_norm = (x - _min) / _range
    return x_norm