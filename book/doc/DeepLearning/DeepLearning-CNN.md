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

 - Pooling layers 池化层

   - 最大池化层

   - 平均池化层

- Fully Connected（FC）全连接层



## 深度CNN







## 目标检测







## 特殊应用