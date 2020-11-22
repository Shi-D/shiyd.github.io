[TOC]

# DeepLearning-CNN

## CNN基础

#### **edge detection**

<img src="/book/doc/DeepLearning/resources/4.png" alt="4" style="zoom:50%;" />

#### **Padding**

用padding进行填充以避免图像边缘像素点的丢失.

p = padding-amount 即每个边缘增加的像素.

n x n为图像原始大小.

f为滤波器的长，通常为**奇数**.

则输出图像长为`（n+2p-f+1）`.

两种常用的padding：

浮动元素和绝对定位元素，非块级盒子的块级容器（例如 inline-blocks, table-cells, 和 table-captions），以及overflow值不为“visiable”的块级盒子，都会为他们的内容创建新的BFC（块级格式上下文）。即：

> 1、Valid Convolutions； 
>
> 2、Same Convolutions； 

Valid Convolutions: `n x n * f x f` -> `(n - f + 1) x (n - f + 1)`

Same Convolutions: `p = (f-1)/2` -> `n x n`

#### **Strided Convolutions(卷积步长)**

s表示卷积步长，则若使用Same Convolutions，输出为 `(n+2p-f)/s+1`

#### **Layers**

* Convolutions
* Pooling layers 池化层
  - 最大池化层
  - 平均池化层

- Fully Connected（FC）全连接层



## 深度CNN

#### classic neural networks

**LeNet-5**

<img src="/book/doc/DeepLearning/resources/LeNet-5.png" alt="LeNet-5"/>

**AlexNet**

<img src="/book/doc/DeepLearning/resources/AlexNet.png" alt="AlexNet"/>

**VGG-16**

包含16个卷积层和全连接层

<img src="/book/doc/DeepLearning/resources/VGG-16.png" alt="VGG-16"/>

#### ResNets(残差网络)

**Residual Block(残差块)**

残差网络为图中紫色path（正常主路径为绿色path），在Relu非线性激活前加上a[l]， a[l]的信息直接到达神经网络的深层，不再沿着主路径传递.

<img src="/book/doc/DeepLearning/resources/ResidualBlock.png" alt="ResidualBlock"/>

残差块能训练更深的神经网络. 所以构建一个ResNet网络就是通过将很多残差块堆积在一起，形成一个深度神经网络.

**Residual Network**

理论上，神经网络越深，训练得更好，但是若没有残差块，随着网络深度的加深，训练错误会越来越多.

**Why ResNets work**

残差学习恒等函数非常容易，可以确定网络性能不会收到影响，很多时候甚至能提高效率.

#### Inception

##### **One by One Conv(Network in Network)**

<img src="/book/doc/DeepLearning/resources/onebyone.png" alt="onebyone"/>



## 目标检测







## 特殊应用