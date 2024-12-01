import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class NonlinearAutoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.weights_encoder = np.random.randn(input_dim, encoding_dim) * 0.01
        self.bias_encoder = np.zeros((1, encoding_dim))
        self.weights_decoder = np.random.randn(encoding_dim, input_dim) * 0.01
        self.bias_decoder = np.zeros((1, input_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        self.encoded = self.relu(np.dot(X, self.weights_encoder) + self.bias_encoder)
        self.decoded = np.dot(self.encoded, self.weights_decoder) + self.bias_decoder
        return self.decoded

    def mse_loss(self, X, output):
        return np.mean((X - output) ** 2)

    def backward(self, X, output, lr):
        error = output - X
        d_weights_decoder = np.dot(self.encoded.T, error) / X.shape[0]
        d_bias_decoder = np.mean(error, axis=0)
        d_encoded = np.dot(error, self.weights_decoder.T)
        d_encoded[self.encoded <= 0] = 0  # ReLU的梯度
        d_weights_encoder = np.dot(X.T, d_encoded) / X.shape[0]
        d_bias_encoder = np.mean(d_encoded, axis=0)

        self.weights_encoder -= lr * d_weights_encoder
        self.bias_encoder -= lr * d_bias_encoder
        self.weights_decoder -= lr * d_weights_decoder
        self.bias_decoder -= lr * d_bias_decoder

def main():
    df_wine = pd.read_csv('data/wine.data', header=None)

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    input_dim = X_train_std.shape[1]
    encoding_dim = 2  # 将数据压缩到2维
    autoencoder = NonlinearAutoencoder(input_dim, encoding_dim)

    epochs = 7000
    lr = 0.01

    for epoch in range(epochs):
        output = autoencoder.forward(X_train_std)
        loss = autoencoder.mse_loss(X_train_std, output)
        autoencoder.backward(X_train_std, output, lr)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    # 使用编码器进行降维
    X_train_encoded = autoencoder.relu(np.dot(X_train_std, autoencoder.weights_encoder) + autoencoder.bias_encoder)

    X_reconstruct = autoencoder.forward(X_train_std)
    reconstruction_error = np.mean((X_train_std - X_reconstruct) ** 2)
    print(f'Reconstruction Error: {reconstruction_error:.4f}')

    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_encoded[y_train == l, 0], X_train_encoded[y_train == l, 1], c=c, label=l, marker=m)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Nonlinear Autoencoder')
    plt.legend(loc='lower left')
    plt.show()

if __name__ == '__main__':
    main()