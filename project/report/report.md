##                    midterm report                   12211810 潘炜
最新版的report和代码链接：https://github.com/WeisonWEileen/homework_sdm274/blob/master/project/report/report.md
****

# Special things I make in the reop
### 作为二分类，0 1样本严重不平衡
### 数据集不均衡问题

7：3 选择测试集和验证集，在样本量为3000的验证集中，只有93个样本```Machine Failure=1```。一种办法是可能是

- 使用 Zhiyun Lin 老师在wechat群提到的在loss上面加 weight
- drop掉一部分 ```Machine Failure=0``` 的样本。
- 使用 data-argumentation 对 Machine Failure 样本进行增强
- 另一种方法是使用 KaiMing He 团队 在 Facebook 时工作的时候发表的
  Focal loss[^1] 来解决这个问题， 其最初是为了解决 foreground-background class imbalance 的不平衡的问题。 （后面有时间再尝试这个方法）

这里由于时间的原因，尝试第二种方法

# Implementation Detail

### data preprocessing

extract data from the csv file, data split the data into training set and test set

![image-20241120110520014](./assets/image-20241120110520014.png)

# feature engineering

### normalization

**Normalization** is an essential preprocessing step in data analysis and machine learning. And since the scale differs with each other greatly， in this repo we use the min-max normalization to normalize the data

### compute the corrolation heatmap

![image-20241120125325338](./assets/image-20241120125325338.png)

sorted by correlation coefficient

![image-20241120125447644](./assets/image-20241120125447644.png)

base on the correlation map

![image-20241120125517753](./assets/image-20241120125517753.png)

hence we choose  ```Torque```, ``` Tool wear```, ```Air temperature``` these 3 feature as our input feature, and ignore other features. Below is the code for feature choose.

![image-20241120132534735](./assets/image-20241120132534735.png)

## Model inplementation

### linear regression

part of code 
![image-20241122223008161](./assets/image-20241122223008161.png)

- inplementation detail:
  - since the samples of the machine failure is unbalanced(about 2907(machine failure = 0) : 97(machine failure = 1) in test set), we need to change the decision boundarydecision boundary .should be set very small ( < 0.3) for this case.
  - remember to normalize the input detail

![image-20241120203059744](./assets/image-20241120203059744.png)

![image-20241120203429538](./assets/image-20241120203429538.png)

### parameter tuning of linear regression

here is the param I run 

![image-20241122222817009](./assets/image-20241122222817009.png)



### perceptron

part of code 

![image-20241122223055518](./assets/image-20241122223055518.png)

we can see that perceptron work with 

![image-20241121170646105](./assets/image-20241121170646105.png)

performance and 4 metrics

![image-20241121170701224](./assets/image-20241121170701224.png)

### parameter tuning of perceptron

![image-20241122223039659](./assets/image-20241122223039659.png)

### logestic regression

code
![image-20241122223157931](./assets/image-20241122223157931.png)

loss

![image-20241121164042985](./assets/image-20241121164042985.png)

4 metrics

![image-20241121164420987](./assets/image-20241121164420987.png)

### paramter tuning

![image-20241122223141142](./assets/image-20241122223141142.png)

### Multiple Layer perceptron

code

![image-20241122223230366](./assets/image-20241122223230366.png)

loss
![image-20241121173732465](./assets/image-20241121173732465.png)

### paramter tuning of MLP

pr 曲线

![image-20241121173506354](./assets/image-20241121173506354.png)

pr 曲线的点对应的metrics

![image-20241122155339842](./assets/image-20241122155339842.png)

**reduce the samples** to 300

样本更加均衡，the loss更快收敛

![image-20241122160237336](./assets/image-20241122160237336.png)

### Model Comparison ： based on best F1 score I get

Among all models, the Logestic regression and MLP model perform the best , however, we need to choose the right threshold to precision and recall balance. The rank of the best model in the parameter I tried:

1. Multiple Layer Perceptron
   0.45

2. Logestric Regression
   0.4

3. Perceptron
   0.37

4. Linera regression
   0.26

   Hence, I MLP perform best on this dataset.



### Reference

[^1]: Lin, T., Goyal, P., Girshick, R.B., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42, 318-327.