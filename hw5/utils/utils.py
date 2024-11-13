import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def count_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)

def filter2(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 过滤掉以 '3' 开头的行
    filtered_lines = [line for line in lines if not line.startswith('3')]

    # 将过滤后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(filtered_lines)

def load_data(file_path):
    # 读取数据
    data = np.loadtxt(file_path, delimiter=',')
    # 分离标签和特征
    y = data[:, 0]  # 第一列是标签
    x = data[:, 1:]  # 其余列是特征

    return y,x

def split_data(x, y, test_size=0.3, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    y_test[y_test == 2] = 0
    y_train[y_train == 2] = 0
    return x_train, x_test, y_train.reshape(-1,1), y_test.reshape(-1,1)

def visualize_fitting_line(x_train, y_train, x_test, y_test, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, s=1, color='blue', label='Training Data')
    plt.plot(
        x_test, model.forward(x_test), color="red", linewidth=2, label="Fitting Line"
    )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Data and Fitting Line')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_blobs(X, y, title="Generated Classification Data"):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=10)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.colorbar(label="Class")
    plt.grid(True)
    plt.show()


def complex_nonlinear_function(x):
    return 10 * np.sin(2 * 3.14 * x) + 0.05 * x**3 + np.random.randn(x.shape[0]) * 0.1


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    print(xx.shape)
    print(yy.shape)
    print(grid.shape)
    Z = np.round(model.forward(grid))
    print(Z.shape)
    # exit()
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=20, cmap="viridis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.colorbar(label="Class")
    plt.grid(True)
    plt.show()
