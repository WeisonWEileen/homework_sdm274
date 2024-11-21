## midterm report

**12211810 潘炜**

### data preprocessing

extract data from the csv file, data split the data into training set and test set

![image-20241120110520014](./assets/image-20241120110520014.png)

### feature engineering

compute the corrolation heatmap

![image-20241120125325338](./assets/image-20241120125325338.png)

sorted by correlation coefficient

![image-20241120125447644](./assets/image-20241120125447644.png)

base on the correlation map

![image-20241120125517753](./assets/image-20241120125517753.png)

hence we choose  ```Torque```, ``` Tool wear```, ```Air temperature``` these 3 feature as our input feature, and ignore other features. Below is the code for feature choose.

![image-20241120132534735](./assets/image-20241120132534735.png)

## Model inplementation

### linear regression

- inplementation detail:
  - since the samples of the machine failure is unbalanced(about 2907(machine failure = 0) : 97(machine failure = 1) in test set), we need to change the decision boundarydecision boundary .should be set very small ( < 0.3) for this case.
  - remember to normalize the input detail

![image-20241120203059744](./assets/image-20241120203059744.png)

![image-20241120203429538](./assets/image-20241120203429538.png)

### perceptron

![image-20241121170646105](./assets/image-20241121170646105.png)

performance and 4 metrics

![image-20241121170701224](./assets/image-20241121170701224.png)



### logestic regression

loss

![image-20241121164042985](./assets/image-20241121164042985.png)

4 metrics

![image-20241121164420987](./assets/image-20241121164420987.png)

### Multiple Layer perceptron

loss![image-20241121173732465](./assets/image-20241121173732465.png)

pr 曲线

![image-20241121173506354](./assets/image-20241121173506354.png)

pr 曲线的点对应的metrics

![image-20241121173801179](./assets/image-20241121173801179.png)
