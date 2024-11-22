import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim, bias=True):
        self.x = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim) * 0.01

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, grad_input):
        batch_size = self.X.shape[0]
        d_b = 1 / batch_size * np.sum(grad_input, axis=0)
        d_W = 1 / batch_size * np.matmul(self.X.T, grad_input)
        grad_output = np.matmul(grad_input, self.W.T)
        return grad_output, d_W, d_b

class Sigmoid:
    def __init__(self):
        self.X = None    

    def forward(self, input):
        self.X = 1 / (1 + np.exp(-input))
        return self.X

    def backward(self, grad_input):
        return grad_input * self.X * (1 - self.X)

class ReLU:
    def __init__(self):
        self.X = None
    
    def forward(self, input):
        self.X = np.maximum(0, input)
        return self.X

    def backward(self, grad_input):
        grad_output = grad_input * (self.X > 0)
        return grad_output

class MSELoss:
    def __init__(self):
        pass

    def forward(self, y_pred, y):
        loss = np.mean((y - y_pred) ** 2)
        return loss
    
    def backward(self, y_pred, y):
        grad_output = 2 * (y_pred - y) / y.shape[0]
        return grad_output
    
class BCE_loss:
    def __init__(self):
        pass

    def forward(self, y_pred, y):
        epsilon = 1e-5
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return loss
    
    def backward(self, y_pred, y):
        epsilon = 1e-5
        grad_output = (y_pred - y) / (epsilon + y_pred * (1 - y_pred))
        return grad_output

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        self.layers = []
        self.activations = []
        
        # Add input layer
        self.layers.append(Linear(input_dim, hidden_dims[0]))
        self.activations.append(self._get_activation(activation))
        
        # Add hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(Linear(hidden_dims[i-1], hidden_dims[i]))
            self.activations.append(self._get_activation(activation))
        
        # Add output layer
        self.layers.append(Linear(hidden_dims[-1], output_dim))
        self.activations.append(Sigmoid())  # Assuming binary classification

    def _get_activation(self, activation):
        if activation == 'relu':
            return ReLU()
        elif activation == 'sigmoid':
            return Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, X):
        for layer, activation in zip(self.layers, self.activations):
            X = layer.forward(X)
            X = activation.forward(X)
        return X

    def backward(self, grad_output):
        grads = []
        for layer, activation in zip(reversed(self.layers), reversed(self.activations)):
            grad_output = activation.backward(grad_output)
            grad_output, d_W, d_b = layer.backward(grad_output)
            grads.append((d_W, d_b))
        grads.reverse()
        return grads

    def update(self, grads, lr):
        for (layer, (d_W, d_b)) in zip(self.layers, grads):
            layer.W -= lr * d_W
            layer.b -= lr * d_b

# 示例使用
if __name__ == "__main__":
    # 创建一个 MLP 实例
    mlp = MLP(input_dim=2, hidden_dims=[4, 4], output_dim=1, activation='relu')
    
    # 生成一些示例数据
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, (10, 1))
    
    # 前向传播
    y_pred = mlp.forward(X)
    
    # 计算损失
    loss_fn = BCE_loss()
    loss = loss_fn.forward(y_pred, y)
    print(f"Loss: {loss}")
    
    # 反向传播
    grad_output = loss_fn.backward(y_pred, y)
    grads = mlp.backward(grad_output)
    
    # 更新参数
    mlp.update(grads, lr=0.01)