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
        self.data = data
        self.depend = depend
        self.name = name
        self.grad = 0
    
    '''
    grad_fn1 操作符左边的梯度
    grad_fn2 操作符右边的梯度
    '''
    def __mul__(self, data):
        """
        left multiply
        y = x * data
        """
        def grad_fn1(grad):
            return grad * data.data
        def grad_fn2(grad):
            return grad * self.data
        new = Tensor(self.data * data.data, depend=[(self, grad_fn1), (data, grad_fn2)])
        return new

    def __rmul__(self, data):
        """
        right multiply
        y = data * x
        """
        def grad_fn1(grad):
            return grad * self.data
        def grad_fn2(grad):
            return grad * data.data
        new = Tensor(self.data * data.data, depend=[(self, grad_fn1), (data, grad_fn2)])
        return new
    
    def __add__ (self, data):
        """
        y = x + data
        """
        def grad_fn(grad):
            return grad
        new = Tensor(self.data + data.data, depend=[(self, grad_fn), (data, grad_fn)])
        return new
    
    def __radd__ (self, data):
        """
        y = x + data
        """
        def grad_fn(grad):
            return grad
        new = Tensor(self.data + data.data, depend=[(self, grad_fn), (data, grad_fn)])
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
        return new

    def __rsub__(self, data):
        """
        y = data - x
        """
        def grad_fn1(grad):
            return -grad
        def grad_fn2(grad):
            return grad
        new = Tensor(data.data - self.data, depend=[(self, grad_fn1), (data, grad_fn2)])
        return new
    
    def __pow__(self, n):
        def grad_fn(grad):
            return grad * n * self.data ** (n-1)
        new = Tensor(self.data ** n, depend=[(self, grad_fn)])
        return new
        
    def backward(self, grad=None):
        if grad == None:
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
    def __init__(self):
        self.weights = Tensor(np.random.randn(1) * 0.01)
        self.bias = Tensor(np.random.randn(1) * 0.01)
    
    def __call__(self, x):
        return self.weights * x + self.bias

def fun(x):
    return 2*x + 1

if __name__ == "__main__":
    x = [2,4,5,6,7,8,9,10,11,12,13]
    y = []
    for i in range(len(x)):
        y.append(fun(x[i]))

    ly = Linear()

    for i in range(len(x)):
        # print(i)
        x_tensor = Tensor(x[i])
        y_tensor = Tensor(y[i])

        loss = (ly(x_tensor) - y_tensor) ** 2
        print(f"for i = {i} loss is {loss.data}")
        loss.backward()

        for param in [ly.weights, ly.bias]:
            param.data -= 0.01 * param.grad
        
        loss.zero_grad()




# x = Tensor(2)
# x2 = x * x
# g = x2 * x2
# h = x2 * x2
# y = g + h
# y.backward()
# print(g.grad)
# print(x.grad)