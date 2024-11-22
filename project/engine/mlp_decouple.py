import numpy as np
import matplotlib.pyplot as plt

class Linear():
    def __init__(self, input_dim, output_dim, bias=True):
        self.x = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim) * 0.01

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, grad_upstream):
        batch_size = self.X.shape[0]
        # same dim with b
        d_b = 1 / batch_size * np.sum(grad_upstream, axis=0)
        # same dim with W
        d_W = 1 / batch_size * np.matmul(self.X.T, grad_upstream)
        # same dim with X
        grad_downstream = np.matmul(grad_upstream,self.W.T)

        return grad_downstream, d_W, d_b

class Sigmoid():
    def __init__(self):
        self.X = None    

    def forward(self, input):
        self.X = 1 / (1 + np.exp(-input))
        return self.X

    def backward(self, grad_upstream):
        return grad_upstream * self.X * (1 - self.X)

class ReLU():
    def __init__(self):
        self.X = None
    
    def forward(self, input):
        self.X = np.maximum(0, input)

    def backward(self, grad_upstream):
        grad_downstream = grad_upstream * (self.X > 0)
        return grad_downstream

class MSELoss():
    def __init__(self):
        pass

    def forward(self, y_pred, y):
        loss = np.mean((y-y_pred)**2)
        return loss
    
    def backward(self, y_pred, y):
        grad_downstream = 2 * (y - y_pred) / y.shape[0]
        return grad_downstream
    
class BCE_loss:
    def __init__(self):
        pass

    def forward(self, y_pred, y):
        epsilon = 1e-5
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1-y)*np.log(1-y_pred + epsilon))
        return y
    
    def backward(self, y_pred, y):
        epsilon = 1e-5
        grad_downstream = (y_pred - y) / (epsilon + y_pred*(1 - y_pred))
        return grad_downstream
    
class MLP:
    def __init__(self, n_feature = 1, n_iter = 200, lr = 1e-3, tol=None):

        self.fc1 = Linear(n_feature, 10)
        self.act1 = ReLU()
        self.fc2 = Linear(10,1)
        self.act2 = Sigmoid()
        self.loss_func = BCE_loss()

        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.loss = []

    def forward(self, X):
        output1 = self.fc1.forward(X)
        output1a = self.act1.forward(output1)
        output2 = self.fc2.forward(output1a)
        output2a = self.act2.forward(output2)
        return output2a
        
    def batch_update(self, X, y):

        for iter in range(self.n_iter):
            y_pred = self.forward(X)
            loss = self.loss_func.forward(y_pred, y)
            self.loss.append(loss)
            if iter % 100 == 0:
                print(f"Epoch: {iter}, Loss: {loss}")

            grad_output2a = self.loss_func.backward(y_pred, y)
            grad_output2 = self.act2.backward(grad_output2a)
            grad_output1a, grad_W2, grad_b2 = self.fc2.backward( grad_output2)
            grad_output1 = self.act1.backward(grad_output1a)
            grad_input, grad_W1, grad_b1 = self.fc1.backward(X, grad_output1)

            self.fc1.W = self.fc1.W - self.lr * grad_W1
            self.fc1.b = self.fc1.b - self.lr * grad_b1
            self.fc2.W = self.fc2.W - self.lr * grad_W2
            self.fc2.b = self.fc2.b - self.lr * grad_b2

    def train(self, X_train, y_train):
        self.batch_update(X_train, y_train)
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.where(y_pred > 0.5, 1, 0)
    
    def plot_loss(self):
        plt.plot(self.loss)
        plt.grid
        plt.show()

if __name__ == "__main__":
    X_train = np.array([[-2,4],[4,1],[1,6],[2,4],[6,2]])
    y_train = np.array([[0],[0],[1],[1],[1]])

    _, n_feature = X_train.shape

    model = MLP(n_feature=n_feature, n_iter=2000, lr=0.1, tol=1.0e-5)
    model.train(X_train, y_train)

    y_pred = model.predict(X_train)
    print(f"Predicted labels are {y_pred}")

    plt.figure()
    model.plot_loss()