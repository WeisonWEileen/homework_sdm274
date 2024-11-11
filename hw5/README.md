# numpy 实现 MLP
### nonlinear function approximation
注意最后一层不要使用 ```sigmoid```
### classifier 
@TODO
参考 pytorch 动态图的计算原理和 aotograd 系统，使用 numpy 实现Multiple Layer Perceptron
### 基本公式
假设输入是 $f_{pred}=f_2(w_2f(w_1x+b_1)+b_2),E=\frac{1}{2}(f_{pred}-f_2)^2$
假设f是```sigmoid```
$$
\partial E/\partial b_2=(f_{pred}-f_2)f_2(1-f_2)\cdot  \\
\partial E/\partial w_2=(f_{pred}-f_2)f_2(1-f_2)\cdot f_1 \\
\partial E/\partial b_1=(f_{pred}-f_2)f_2(1-f_2)\cdot w_2 \cdot f_1(1-f_1)\\
\partial E/\partial b_2=(f_{pred}-f_2)f_2(1-f_2)\cdot w_2 \cdot  f_1(1-f_1) \cdot x
$$



# n_units
第一个数表示 input_dim
中间表示 hidden layer 的 units
最后一个数表示 output_dim
# dimmesion issue
第$i-1$层有m个unit，第$i$层有n个unit，那么第$i$层的权重矩阵的维度是$n \times m$，而不是$m \times n$，这样可以方便的进行矩阵乘法运算。
线性部分对应的乘法就是 ```np.dot(w[i],X.T)```
# tips
- 设计矩阵乘法的操作的时候, np.dot 或者是 np.multiply 决定是否使用逐元素相乘
- 选用 .T (不需要使用reshape) 即可维持运算过程中矩阵维度的正确性
- 每行表示一个样本
- 对于多样本的计算，每一个样本有一个单独的梯度流作用到某个neuron。梯度没传播到一层就进行一个 ```axis=1``` 的求和。每个```neuron```上就保留了每个梯度上的dz信息。
- batch backward 的过程中，对于 bias 的梯度，需要将所有的行求平均来均分梯度。而对于 weights，对 ```a[i]```转置之后，就会自动刚好将梯度求和。所以只会出现一次 ```np.sum(..., axis=[0], keepDim=True)```

### run scripts
configure options in ```conf/config.yaml```  
and run
```
cd hw3
python3 -m scripts.train
```
### @TODO
- 参考```pytorch```将代码分解成操作子，实现 autograd

### 矩阵乘法维度选取
对于某个样本点，我们一般习惯使用**行向量**去表示。这意味着我们的输入矩阵的维度是$X_{(n\times m)}$。如果使用$W\cdot X^T$,会发现需要再取一次转置才能使得中间层的 units 变回行向量的形式（unints的相对于下一层来说就是一个正常的样本）。于是我们可以直接使用$X\cdot W$，这样就可以直接得到中间层的输出。这样的好处是可以直接使用矩阵乘法的形式，而不需要再次转置。即
$$
y^{(i)}=\sum_{j=0}^nw_jx_j^{(i)}=w^\top x^{(i)}\Rightarrow y=\boldsymbol{X}\boldsymbol{w}
$$
#### 参考
https://github.com/zinsmatt/Neural-Network-Numpy/blob/master/neural-network.py
https://zhuanlan.zhihu.com/p/501743440

### 有点矛盾，就是
- X最好一个横向量表示一个维度
- 但是要实现矩阵乘法 中间层就是列向量