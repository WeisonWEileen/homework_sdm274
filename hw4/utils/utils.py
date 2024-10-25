import numpy as np
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
