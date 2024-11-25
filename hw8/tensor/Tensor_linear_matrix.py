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

if __name__ == "__main__":
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


