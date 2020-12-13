

[TOC]



# Python的一些小tricks

## 数据处理

#### np.c、np.r_

`np.c_`是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。

```python
a = np.array([[1, 2, 3],[7,8,9]])
b=np.array([[4,5,6],[1,2,3]])
c=np.c_[a,b]

c
Out[2]: 
array([[1, 2, 3, 4, 5, 6],
       [7, 8, 9, 1, 2, 3]])
```



`np.r_`是把两矩阵上下相加，要求列数相等。

```python
a = np.array([[1, 2, 3],[7,8,9]])
b=np.array([[4,5,6],[1,2,3]])
 
d= np.array([7,8,9])
e=np.array([1, 2, 3])
 
g=np.r_[a,b]
 
g
Out[14]: 
array([[1, 2, 3],
       [7, 8, 9],
       [4, 5, 6],
       [1, 2, 3]])
 
h=np.r_[d,e]
 
h
Out[16]: array([7, 8, 9, 1, 2, 3])
```



#### ravel()、flatten()、squeeze()、reshape()

将多维数组转换为一维数组

```python
arr = np.range(12).reshape(4,3)

arr
Out[17]:
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9],
       [10, 11, 12]])
```

```python
a = arr.ravel()

a
Out[18]:
arr([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

```python
b = arr.flatten()

b
Out[19]:
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

```python
c = arr.reshape(-1)

c
Out[20]:
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

```python
d = arr.squeeze()

d
Out[21]:
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```



#### zip()

打包

```python
for input, label in zip(inputs, labels):
    pass
```

```python
a = [[1,2], [2,3], [3,4]]
b = [7,8,9]
c = zip(a, b)
print(c)

for i, j in c:
    print(i, j)
    
<zip object at 0x1082f1cc8>
[1, 2] 7
[2, 3] 8
[3, 4] 9
```



## matplotlib绘图

#### cmap取值

```python
cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']

cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
```

参考matploylib文档：https://matplotlib.org/tutorials/colors/colormaps.html?highlight=rdylgn



#### 用雷达图绘制二维坐标分类问题

```python
x_min, x_max = -5, 5
y_min, y_max = -5, 5
#h越小，精确度越高
h = 0.01
# 先搞定网格，横纵坐标
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 将横纵坐标拼接起来，即为x_test
x_test = np.c_[xx.ravel(), yy.ravel()].T
print(x_test.T)
#将（-5，5）范围内的点放入模型进行预测
Z = predict(torch.Tensor(x_test.T))
print(Z.shape)
Z = Z.reshape(xx.shape)
print(Z.shape)
# 绘制雷达图，cmap取值见上面的知识点
plt.contourf(xx, yy, Z, cmap='RdYlGn')
#绘制散点图做比对
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.squeeze(), cmap='RdYlGn')
```



## 其他

代码开头

```
#!/usr/bin/env python
# -*- coding:utf-8 -*-
```

DL常用package

```python
import torch
import torch.nn as nn
from torch.utils.data import dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
```

