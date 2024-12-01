# 12211810 潘炜 AI and Machine Learning hw 9

code: https://github.com/WeisonWEileen/homework_sdm274/tree/master/hw9

you can change the param in ```config.yaml``` to reproduce all the results metioned in the report.

### PCA \

code for the pca

![image-20241201204622680](./assets/image-20241201204622680.png)

we consider the two component, which is 

![image-20241201204326424](./assets/image-20241201204326424.png)

then we reconstruct it, and visualize the result 	

![image-20241201204033602](./assets/image-20241201204033602.png)

reconstruction error:

![image-20241201212946523](./assets/image-20241201212946523.png)

### Linear AutoEncoder

![image-20241201212312664](./assets/image-20241201212312664.png)



![image-20241201212024462](./assets/image-20241201212024462.png)

reconstruction error:

![image-20241201214621797](./assets/image-20241201214621797.png)

### None-Linear Auto-encoder

使用一层 Relu 作为 activation function

![image-20241201214415487](./assets/image-20241201214415487.png)

![image-20241201214314375](./assets/image-20241201214314375.png)

recontruction error

![image-20241201214439907](./assets/image-20241201214439907.png)

### Comparison and Analysis

- error : pca = linear autoencoder < nonlinear autoencoder

- 由于所有维度的数据均是做了**去中心化**的处理。这意味着：只要训练的超参数合理，PCA 和 linear Autoencoder 是等价的，而上面跑的代码得到的reconstruction error也是一致的. 而 带一层 activation function 为 relu 的结果稍逊，但是也没有差到哪里去。pca和linear ate 比 none-linear autoencoder 更加好的原因可能是，这个数据集的数据更加偏向**线性可分**