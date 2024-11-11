# AI and Machine Learning homework 05 MLP

### 12211810 潘炜

code: https://github.com/WeisonWEileen/homework_sdm274/tree/master/hw5
you can change the param in ```config.yaml``` to reproduce all the results metioned below.

### MLP

![image-20241111191201053](./assets/image-20241111191201053.png)

Multiple Layer Perceptron is a classical deep learning model, based on universal approximation theory, it can approximate any function as its depth increase.

# Nonelinear fitting

### nonelinear function

![image-20241111193502197](./assets/image-20241111193502197.png)

### hyperparameters

![image-20241111192017374](./assets/image-20241111192017374.png)

### Plotting results of 3 different layers trys

I try 3 different layers, and plot the output here.  

![image-20241111191628743](./assets/image-20241111191628743.png)

the final loss 

![image-20241111194559938](./assets/image-20241111194559938.png)

### Conclusion

- ,Adding more layers (depth) or neurons (width) increases the model’s capacity to capture intricate nonlinear patterns.

- some times we need to consider the computational loss. The more complex the model is, the more computation time it takes.

# Classfier

generate dataset helped by **ChatGPT**

![image-20241111194241895](./assets/image-20241111194241895.png)

layer:```[2,10,1]```: final loss :0.05125819234234

 ![image-20241111192405627](./assets/image-20241111192405627.png)

layer:[1,17,1]: final loss : 0.04129081951076237

![image-20241111193016216](./assets/image-20241111193016216.png)

layer:[2,10,20,1] final: loss 0.04296898704408965

![image-20241111193212321](./assets/image-20241111193212321.png)

loss plotting

![image-20241111193539472](./assets/image-20241111193539472.png)

at the last test, we compute 4 metrics

![image-20241111193647420](./assets/image-20241111193647420.png)

### Conclusion

- when increase model complexity, the boundry may be more curve, that may be overfitting

- The MLP’s hidden layers with nonlinear activation functions allow it to model complex, nonlinear decision boundaries that separate different classes effectively, which is beneficial for datasets that are not linearly separable.