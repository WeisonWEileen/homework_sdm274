## 使用纯 numpy 构建 autograd
autograd 机制能够完美的将神经网络中的算子的 Module 性质结合，这里将使用numpy完全手写autograd系统，并且添加一下的经典的神经网络的示例。
### nerual network example
- MLP
- CNN
- RNN
- LSTM
- Transformer

### @TODO
- 提供打印 computation graph 的接口
- 构建拓扑关系
- 加上权重 penalty

###  实现的细节

注意，这么写是错的，应该是使用下一级的值，而不是 self.data（这只是 z）

![image-20241125212238032](./assets/image-20241125212238032.png)
