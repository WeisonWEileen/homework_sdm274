import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据集
file_path = './data/ai4i2020.csv'
data = pd.read_csv(file_path)

# 查看数据集的前几行
print(data.head())

# 删除无用的列（第一列和第二列）
data = data.drop(columns=['UDI', 'Product ID', 'HDF', 'OSF', 'PWF', 'TWF', 'RNF'])

# 将第三列 'Type' 编码为数值：M -> 0, L -> 1
data['Type'] = np.where(data['Type'] == 'M', 0, 1)

# 计算相关性矩阵
correlation_matrix = data.corr()

# 绘制相关性热图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 提取与 Machine failure 相关的特征
correlation_with_failure = correlation_matrix['Machine failure'].sort_values(ascending=False)
print(correlation_with_failure)