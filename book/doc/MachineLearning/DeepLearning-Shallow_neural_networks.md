[TOC]

# DeepLearning-Shallow neural networks

**Neural Network Representation**

<img src="/book/doc/DeepLearning/resources/1.png" alt="4" style="zoom:90%;" />

**Activation functions**

<img src="/book/doc/DeepLearning/resources/2.png" alt="4" style="zoom:30%;" />

**Random Initialization**

初始化神经网络的权重，需要随机初始化而非置0. 否则梯度下降将不会起作用. 

一般权重设较小值，使用sigmoid、tanh时，可使z位于函数的

<img src="/book/doc/DeepLearning/resources/3.png" alt="4" style="zoom:50%;" />

如图红色部分，梯度下降效率高. 