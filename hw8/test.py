import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None  # 梯度
        self._backward = lambda: None  # 反向传播函数
        self._prev = set()  # 上游节点

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        visited = set()
        topo_order = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo_order.append(t)

        build_topo(self)

        for t in reversed(topo_order):
            t._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + other.data * out.grad
            if other.requires_grad:
                other.grad = other.grad + self.data * out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other ** -1

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + np.ones_like(self.data) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    def mean(self):
        return self.sum() / self.data.size

    @staticmethod
    def tanh(x):
        t = np.tanh(x.data)
        out = Tensor(t, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                x.grad = x.grad + (1 - t ** 2) * out.grad

        out._backward = _backward
        out._prev = {x}
        return out

class Linear:
    def __init__(self, in_features, out_features):
        self.weights = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.weights + self.bias

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def __call__(self, x):
        x = Tensor.tanh(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":
    # 数据生成
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] * 2 + X[:, 1] * -3 + 1 > 0).astype(np.float32)[:, None]

    # 模型定义
    mlp = MLP(input_size=2, hidden_size=10, output_size=1)

    # 超参数
    epochs = 1000
    lr = 0.01

    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        inputs = Tensor(X)
        targets = Tensor(y)
        predictions = mlp(inputs)
        loss = ((predictions - targets) ** 2).mean()

        # 反向传播
        loss.backward()

        # 参数更新
        for param in [mlp.fc1.weights, mlp.fc1.bias, mlp.fc2.weights, mlp.fc2.bias]:
            param.data -= lr * param.grad
            param.grad = None  # 重置梯度

        # 打印损失
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
