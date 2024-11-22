# 本项目旨在学习使用一些比较现代的机器学习、深度学习配置框架
- 使用 ```omegaconf```和```hydra```库进行参数配置 

# install dependencies
```
pip install numpy hydra-core omegaconf wandb  
```
# 框架
### Finished
- 使用 decorator 配置多种作业要求
- hydra参数配置框架配置
### @ TODO 
- wandb loss 可视化loss, 配置 gradient 和 loss 的可视化的接口
- 参考老师课件改善 mlp
- 对照 pytorch 的写法。对比一下。
- 样本太多了？ bce loss 在
- 样本的不均的问题，看看可不可以使用 [focal loss](https://www.notion.so/SDM-274-11103fe86de580849b2cef37ef887f7b) 来制约一下？
