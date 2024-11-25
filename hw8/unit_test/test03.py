import numpy as np
from sklearn.model_selection import train_test_split

class Tensor():
    def __init__(self, data, depend=[], name="none"):
        self.data = np.array(data, dtype=np.float32)
        self.depend = depend
        self.name = name
        self.grad = np.zeros_like(self.data)
    
    def __matmul__(self, data):
        def grad_fn1(grad):
            return np.matmul(grad, data.data.T)
        def grad_fn2(grad):
            return np.matmul(self.data.T, grad)
        new = Tensor(np.matmul(self.data, data.data), depend=[(self, grad_fn1), (data, grad_fn2)])
        new.name = "mul of " + self.name + " and " + data.name
        return new
    
    def __add__ (self, data):
        def grad_fn(grad):
            return grad
        new = Tensor(self.data + data.data, depend=[(self, grad_fn), (data, grad_fn)])
        new.name = "add of " + self.name + " and " + data.name
        return new
    
    def relu(self):
        def grad_fn(grad):
            return grad * (self.data > 0)
        a = np.maximum(0, self.data)
        new = Tensor(a, depend=[(self, grad_fn)])
        new.name = "relu of " + self.name
        return new

    def softmax(self):
        def grad_fn(grad):
            return grad * self.data * (1 - self.data)
        exp_data = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        a = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        new = Tensor(a, depend=[(self, grad_fn)])
        new.name = "softmax of " + self.name
        return new
    
    def cross_entropy_loss(self, groundtruth):
        def grad_fn(grad):
            return grad * (self.data - groundtruth.data)
        epsilon = 1e-12
        predictions_clipped = np.clip(self.data, epsilon, 1 - epsilon)
        loss_data = -np.sum(groundtruth.data * np.log(predictions_clipped), axis=1)
        loss = Tensor(loss_data.mean(), depend=[(self, grad_fn)])
        loss.name = "cross_entropy_loss"
        return loss

    def backward(self, grad=None):
        if grad is None:
            self.grad = 1
            grad = 1
        else:
            self.grad += grad
        for tensor, grad_fn in self.depend:
            bw = grad_fn(grad)
            tensor.backward(bw)
    
    def zero_grad(self):
        self.grad = 0
        for tensor, grad_fn in self.depend:
            tensor.zero_grad()

class Linear():
    def __init__(self, shape, batchsize, name):
        self.weights = Tensor(np.random.randn(shape[0], shape[1]) * 0.01, name=name + "weights")
        self.bias = Tensor(np.zeros((1, shape[1])), name=name + "bias")
        self.bias.grad = np.zeros((batchsize, shape[1]))
    
    def forward(self, x):
        return x @ self.weights + self.bias

class MLP():
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        self.fc1 = Linear((input_size, hidden_size), batch_size, "fc1")
        self.fc2 = Linear((hidden_size, output_size), batch_size, "fc2")
    
    def forward(self, x):
        x = self.fc1.forward(x).relu()
        x = self.fc2.forward(x).softmax()
        return x

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

def main():
    X_train, y_train = load_data('../data/optdigits.tra')
    X_test, y_test = load_data('../data/optdigits.tes')

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    # print(y_train[10])
    # exit()

    batch_size = X_train.shape[0]
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 10

    mlp = MLP(input_size, hidden_size, output_size, batch_size)

    epochs = 1000
    lr = 0.01

    for epoch in range(epochs):
        inputs = Tensor(X_train, name="inputs")
        targets = Tensor(y_train, name="targets")
        predictions = mlp.forward(inputs)
        loss = predictions.cross_entropy_loss(targets)
        loss.backward()

        mlp.fc1.weights.data -= lr * mlp.fc1.weights.grad
        mlp.fc1.bias.data -= lr * np.sum(mlp.fc1.bias.grad, axis=0, keepdims=True)
        mlp.fc2.weights.data -= lr * mlp.fc2.weights.grad
        mlp.fc2.bias.data -= lr * np.sum(mlp.fc2.bias.grad, axis=0, keepdims=True)
        loss.zero_grad()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

    inputs = Tensor(X_test, name="inputs")
    targets = Tensor(y_test, name="targets")
    predictions = mlp.forward(inputs)
    accuracy = np.mean(np.argmax(predictions.data, axis=1) == np.argmax(targets.data, axis=1))
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()