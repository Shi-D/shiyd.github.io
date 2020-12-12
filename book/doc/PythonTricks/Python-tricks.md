

# Python的一些小tricks

## 数据处理

#### np.c、np.r_

##### `np.c_`是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。

```python
a = np.array([[1, 2, 3],[7,8,9]])
b=np.array([[4,5,6],[1,2,3]])
c=np.c_[a,b]

c
Out[2]: 
array([[1, 2, 3, 4, 5, 6],
       [7, 8, 9, 1, 2, 3]])
```



##### `np.r_`是把两矩阵上下相加，要求列数相等。

```
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

```
arr = np.range(12).reshape(4,3)

arr
Out[17]:
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9],
       [10, 11, 12]])
```

```
a = arr.ravel()

a
Out[18]:
arr([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

```
b = arr.flatten()

b
Out[19]:
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

```
c = arr.reshape(-1)

c
Out[20]:
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

```
d = arr.squeeze()

d
Out[21]:
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```





## matplotlib绘图

#### cmap取值

```
cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']

cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
```

参考matploylib文档：https://matplotlib.org/tutorials/colors/colormaps.html?highlight=rdylgn



