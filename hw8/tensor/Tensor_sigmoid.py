import numpy as np

class Tensor():
    """
    basic: nodes in comuattional Graph
    """
    def __init__(self, data, depend=[], name="none"):
        """
        data: node name
        depend: node's parents
        name: node name
        """
        self.data = np.array(data, dtype=np.float32)
        self.depend = depend
        self.name = name
        self.grad = np.zeros_like(self.data)
    
    '''
    grad_fn1 操作符左边的梯度
    grad_fn2 操作符右边的梯度
    '''
    def __matmul__(self, data):
        """
        left multiply
        y = X * DATA
        """
        def grad_fn1(grad):
            grad_out = np.matmul(grad, data.data.T)
            return grad_out
        def grad_fn2(grad):
            grad_out = np.matmul(self.data.T, grad)
            # print shape of them 
            # print(self.data.T.shape)
            # print(grad.shape)
            # print("grad_fn2", grad_out.shape)
            return grad_out
        new = Tensor(np.matmul(self.data, data.data), depend=[(self, grad_fn1), (data, grad_fn2)])
        new.name = "mul of " + self.name + " and " + data.name
        return new
    
    def __add__ (self, data):
        """
        y = x + data
        """
        def grad_fn(grad):
            return grad
        new = Tensor(self.data + data.data, depend=[(self, grad_fn), (data, grad_fn)])
        new.name = "add of " + self.name + " and " + data.name
        return new
    
    def __sub__(self, data):
        """
        y = x - data
        """
        def grad_fn1(grad):
            return grad
        def grad_fn2(grad):
            return -grad
        new = Tensor(self.data - data.data, depend=[(self, grad_fn1), (data, grad_fn2)])
        new.name = "sub of " + self.name + " and " + data.name
        return new
    
    def __pow__(self, n):
        def grad_fn(grad):
            return grad * n * self.data ** (n-1)
        new = Tensor(self.data ** n, depend=[(self, grad_fn)])
        new.name = "power of " + self.name
        return new
    
    # def sum(self):
    #     def grad_fn(grad):
    #         return np.ones_like(self.data) * grad
    #     new = Tensor(self.data.sum(), depend=[(self, grad_fn)])
    #     new.name = "sum of " + self.name
    #     return new
    
    # def mean(self):
    #     '''
    #     mean = 1 / batch_size * sum
    #     '''
    #     def grad_fn(grad):
    #         return np.ones_like(self.data) * grad
    #     new = Tensor(self.data.sum(), depend=[(self, grad_fn)])
    #     new.name = "sum of " + self.name

    def average(self):
        '''
        average = self.data.sum() / self.data.size
        '''
        def grad_fn(grad):
            return 1 / self.data.shape[0] * np.ones_like(self.data) * grad
        new = Tensor(self.data.sum() / self.data.shape[0], depend=[(self, grad_fn)])
        new.name = "average of " + self.name
        return new  
    
    # @staticmethod
    def sigmoid(self):
        """
        y = 1 / (1 + exp(-x))
        """
        def grad_fn(grad):
            return grad * self.data * (1 - self.data)
        a = 1 / (1 + np.exp(-self.data))
        new = Tensor(a, depend=[(self, grad_fn)])
        new.name = "sigmoid of " + self.name
        return new

    def bce_loss(predictions, groundtruth):
        """
        Binary Cross-Entropy Loss
        loss = - (y * log(p) + (1 - y) * log(1 - p))
        """
        def grad_fn(grad):
            grad_pred = grad * (predictions.data - groundtruth.data) / (predictions.data * (1 - predictions.data))
            return grad_pred
        
        epsilon = 1e-12  # 防止 log(0)
        predictions_clipped = np.clip(predictions.data, epsilon, 1 - epsilon)
        loss_data = - (groundtruth.data * np.log(predictions_clipped) + (1 - groundtruth.data) * np.log(1 - predictions_clipped))
        loss = Tensor(loss_data.mean(), depend=[(predictions, grad_fn)])
        loss.name = "bce_loss"
        return loss
    

    def backward(self, grad=None):
        # print("back in", self.name)
        if grad is None:
            self.grad = 1
            grad = 1
        else:
            # 多个连接点的叠加
            self.grad += grad
        for tensor, grad_fn in self.depend:
            bw = grad_fn(grad)
            tensor.backward(bw)
    
    def zero_grad(self):
        self.grad = 0
        for tensor, grad_fn in self.depend:
            tensor.zero_grad()

class Linear():
    def __init__(self,shape,batchsize):
        self.weights = Tensor(np.random.randn(shape[0],shape[1]) * 0.01, name="LinearLayer1_weights")
        self.bias = Tensor(np.zeros((1, shape[1])),name="LinearLayer1_bias")
        # grad.shape[0] 扩张成和 batch_size 一样大
        self.bias.grad = np.zeros((batchsize, shape[1]))
    
    def forward(self, x):
        return x @ self.weights + self.bias

def fun(x):
    return 2*x + 1


def main():
    # 数据生成
    np.random.seed(42)
    X = np.random.randn(60, 3)  # 10个样本，每个样本有3个特征
    y = (X[:, 0] * 2 + X[:, 1] * -3 + X[:, 2] * 1 + 1 > 0).astype(np.float32)[:, None]
    # 模型定义
    batch_size = X.shape[0]
    input_size = X.shape[1]
    output_size = 1
    linear_layer = Linear((input_size, output_size), batch_size)

    # 超参数
    epochs = 8000
    lr = 0.01

    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        inputs = Tensor(X, name="inputs")
        targets = Tensor(y, name="targets")
        predictions = linear_layer.forward(inputs)
        predictions = Tensor.sigmoid(predictions)
        # loss = ((predictions - targets) ** 2).average()
        loss = predictions.bce_loss(targets)
        # 反向传播
        loss.backward()

        linear_layer.weights.data -= lr * linear_layer.weights.grad

        linear_layer.bias.data -= lr * np.sum(linear_layer.bias.grad, axis=0, keepdims=True)
        loss.zero_grad()

        # 打印损失
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data:.4f}")

    x_test = Tensor(np.array([[3,2,-1]]))
    y = linear_layer.forward(x_test)
    y = Tensor.sigmoid(y)
    print(y.data)

if __name__ == "__main__":
    main()
    exit()
    x = np.array([
        [2,3,4],
        [5,6,7],
        [8,9,10],
        [1,2,3],
        [3,2,1]
    ])
    #  依次乘 2 乘4 乘 6
    y = np.zeros((x.shape[0], 1))
    
    for i in range(x.shape[0]):
        y[i][0] = 2*x[i][0] + x[i][1] + x[i][2]
    
    ly = Linear((x.shape[1],y.shape[1]), batchsize = x.shape[0])

    for i in range(100):
        # print(i)
        x_tensor = Tensor(x, name="input_x")
        y_tensor = Tensor(y, name="input_y")

        y_pred = ly.forward(x_tensor)
        loss = (y_pred - y_tensor) ** 2
        loss.name = "loss"
        print(f"for i = {i} loss is {loss.data}")
        loss.backward()

        # for param in [ly.weights, ly.bias]:
            # param.data -= 0.01 * param.grad
        ly.weights.data -= 0.001 * ly.weights.grad
        ly.bias.data -= 0.001 * np.sum(ly.bias.grad, axis=0, keepdims=True)
        loss.zero_grad()

    y_pred = ly.forward(Tensor(np.array([[-1,2,3]])))
    print(y_pred.data)


