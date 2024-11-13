# AI and Machine Learning homework 05 MLP

### 12211810 潘炜

code: https://github.com/WeisonWEileen/homework_sdm274/tree/master/hw5

online report (click to check the latest report I submiit):

you can change the param in ```config.yaml``` to reproduce all the results metioned below.

### MLP

![image-20241111191201053](./assets/image-20241111191201053.png)

Multiple Layer Perceptron is a classical deep learning model, based on universal approximation theory, it can approximate any function as its depth increase.

# Nonelinear fitting

### nonelinear function

![image-20241112225657562](./assets/image-20241112225657562.png)

## Mini- Batch 

## hyperparameters

![image-20241112225837191](./assets/image-20241112225837191.png)

![image-20241112225903837](./assets/image-20241112225903837.png)

final loss ![image-20241112225937977](./assets/image-20241112225937977.png)

## SGD

![image-20241112230917580](./assets/image-20241112230917580.png)

### ![image-20241112230940899](./assets/image-20241112230940899.png)

![image-20241112230952632](./assets/image-20241112230952632.png)

### Conclusion

- ,Adding more layers (depth) or neurons (width) increases the model’s capacity to capture intricate nonlinear patterns.

- some times we need to consider the computational loss. The more complex the model is, the more computation time it takes.



### using K-fold Cross validation  to find the best hyperparameter sets

![image-20241112232328820](./assets/image-20241112232328820.png)

the runing best set 

![image-20241112232738726](./assets/image-20241112232738726.png)

### Conclusion

the deeper the network doesn't mean that the better it is, 而且神经网络并不是都是具有可解释性的，我们需要试出来最好的结构

# Classfier

### generate dataset using method ```make_blobs``` in  ```scikit-learn``` 

![image-20241114013825397](./assets/image-20241114013825397.png)

### run configuration

![image-20241114013046689](./assets/image-20241114013046689.png)

### all run results runing using k_fold = 5

![image-20241114013117588](./assets/image-20241114013117588.png)

the best configuration is   ```units=[2,5,10,1], lr=0.01``` 

at the best test, we compute 4 metrics

![image-20241111193647420](./assets/image-20241111193647420.png)



### Decision boundry visualization

![image-20241114013632640](./assets/image-20241114013632640.png)

### Conclusion

- when increase model complexity, the boundry may be more curve, that may be overfitting
- The MLP’s hidden layers with nonlinear activation functions allow it to model complex, nonlinear decision boundaries that separate different classes effectively, which is beneficial for datasets that are not linearly separable.

### things to improve

- change the loss function to cross entropy loss function

  