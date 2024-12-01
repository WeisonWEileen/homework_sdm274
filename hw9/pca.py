import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

def main():

    df_wine = pd.read_csv('data/wine.data', header=None)

    X,y =df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    cov_mat = np.cov(X_train_std.T)
    eigen_vals, engen_vecs = np.linalg.eig(cov_mat)

    tot = sum(eigen_vals)

    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    print(var_exp)

    cum_var_exp = np.cumsum(var_exp)


    
    eigen_pairs = [(np.abs(eigen_vals[i]), engen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    print("eigen value of 2 ")
    print(eigen_pairs[0][0],eigen_pairs[1][0])
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    print("Matrix W:\n", w)
    x_train_pca = X_train_std.dot(w)

    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(x_train_pca[y_train == l, 0], x_train_pca[y_train == l, 1], c=c, label=l, marker=m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    X_train_reconstructed = x_train_pca.dot(w.T)

    # 计算重构误差
    reconstruction_error = np.mean((X_train_std - X_train_reconstructed) ** 2)
    print(f'Reconstruction Error: {reconstruction_error:.4f}')



if __name__ == '__main__':
    main()
