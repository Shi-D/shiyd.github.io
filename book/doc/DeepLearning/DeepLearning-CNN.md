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

<img src="/book/doc/DeepLearning/resources/LeNet-5.png" alt="LeNet-5" style="zoom:60%;" />

**AlexNet**

<img src="/book/doc/DeepLearning/resources/AlexNet.png" alt="AlexNet" style="zoom:60%;"/>

**VGG-16**

包含16个卷积层和全连接层

<img src="/book/doc/DeepLearning/resources/VGG-16.png" alt="VGG-16" style="zoom:60%;"/>

#### ResNets(残差网络)

**Residual Block(残差块)**

残差网络为图中紫色path（正常主路径为绿色path），在Relu非线性激活前加上a[l]， a[l]的信息直接到达神经网络的深层，不再沿着主路径传递.

<img src="/book/doc/DeepLearning/resources/ResidualBlock.png" alt="ResidualBlock" style="zoom:60%;"/>

残差块能训练更深的神经网络. 所以构建一个ResNet网络就是通过将很多残差块堆积在一起，形成一个深度神经网络.

**Residual Network**

理论上，神经网络越深，训练得更好，但是若没有残差块，随着网络深度的加深，训练错误会越来越多.

**Why ResNets work**

残差学习恒等函数非常容易，可以确定网络性能不会收到影响，很多时候甚至能提高效率.

#### Inception

**One by One Conv(Network in Network)**

<img src="/book/doc/DeepLearning/resources/onebyone.png" alt="onebyone" style="zoom:40%;"/>

**Inception Module**

<img src="/book/doc/DeepLearning/resources/inception_module.PNG" alt="Inception_Module" style="zoom:25%;"/>

**Inception Network**

<img src="/book/doc/DeepLearning/resources/inception_network.PNG" alt="Inception_Network" style="zoom:40%;"/>

#### 迁移学习

举个例子，假如说要建立一个猫咪检测器，用来检测自己的宠物猫。比如网络上的**Tigger**，是一个常见的猫的名字，**Misty**也是比较常见的猫名字。假如我的两只猫叫**Tigger**和**Misty**，还有一种情况是，两者都不是。所以现在有一个三分类问题，图片里是**Tigger**还是**Misty**，或者都不是（忽略两只猫同时出现在一张图片里的情况）。现在没有**Tigger**或者**Misty**的大量的图片，所以训练集会很小，该怎么办呢？

这时，可以从网上下载一些神经网络开源的实现，不仅把代码下载下来，也把权重下载下来。有许多训练好的网络，都可以下载。举个例子，**ImageNet**数据集，它有1000个不同的类别，因此这个网络会有一个**Softmax**单元，它可以输出1000个可能类别之一。

但是我只需要识别是不是我的Tigger或者Misty或者都不是，这只是个三分类问题，这时就可以去掉**ImageNet**数据集的**Softmax**层，创建自己的**Softmax**单元，用来输出**Tigger**、**Misty**和**neither**三个类别。

通过使用其他人预训练的权重，即使只有一个小的数据集，也有很大可能可以得到很好的性能。幸运的是，大多数深度学习框架都支持这种操作，事实上，取决于用的框架，它也许会有`trainableParameter=0`这样的参数，对于这些前面的层，可能会需要设置这个参数。为了不训练这些权重，有时也会有`freeze=1`这样的参数。不同的深度学习编程框架有不同的方式，允许你指定是否训练特定层的权重。在这个例子中，你只需要训练**softmax**层的权重，把前面这些层的权重都冻结即可。

<img src="/book/doc/DeepLearning/resources/Transfer_Learning.png" alt="Transfer_Learning" style="zoom:35%;"/>

另一个技巧，也许对一些情况有用，由于前面的层都冻结了，相当于一个固定的函数，不需要改变。因为你不需要改变它，也不训练它，取输入图像，然后把它映射到这层（**softmax**的前一层）的激活函数。所以这个能加速训练的技巧就是，如果我们先计算这一层（紫色箭头标记），计算特征或者激活值，然后把它们存到硬盘里。你所做的就是用这个固定的函数，在这个神经网络的前半部分（**softmax**层之前的所有层视为一个固定映射），取任意输入图像，然后计算它的某个特征向量，这样你训练的就是一个很浅的**softmax**模型，用这个特征向量来做预测。对你的计算有用的一步就是对你的训练集中所有样本的这一层的激活值进行预计算，然后存储到硬盘里，然后在此之上训练**softmax**分类器。所以，存储到硬盘或者说预计算方法的优点就是，你不需要每次遍历训练集再重新计算这个激活值了。



## 目标检测







## 特殊应用