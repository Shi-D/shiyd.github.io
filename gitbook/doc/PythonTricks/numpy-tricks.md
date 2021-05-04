[TOC]

# numpy-tricks

## sp.coo_matrix\csr\csc

#### 1 sp.coo_matrix

```python
import scipy.sparse as sp

>>> sp.coo_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)
       
>>> row = np.array([0, 3, 1, 0])
>>> col = np.array([0, 3, 1, 2])
>>> data = np.array([6, 5, 7, 8])
>>> sp.coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
array([[6, 0, 8, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])
# 0排0列：6

```



#### 2 sp.csr_matrix

按row行来压缩 

```python
from scipy.sparse import *
 
row =  [0,0,0,1,1,1,2,2,2]#行指标
col =  [0,1,2,0,1,2,0,1,2]#列指标
data = [1,0,1,0,1,1,1,1,0]#在行指标列指标下的数字
team = csr_matrix((data,(row,col)),shape=(3,3))
print(team)
print(team.todense())
 
 
输出结果：
  (0, 0)	1
  (0, 1)	0
  (0, 2)	1
  (1, 0)	0
  (1, 1)	1
  (1, 2)	1
  (2, 0)	1
  (2, 1)	1
  (2, 2)	0
[[1 0 1]
 [0 1 1]
 [1 1 0]]
 
Process finished with exit code 0
```

```python
>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> sp.csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])
```



#### 3 sp.csc_matrix

按col列来压缩 

```
>>> sp.csc_matrix((3, 4), dtype=np.int8).toarray()
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=int8)

>>> row = np.array([0, 2, 2, 0, 1, 2])
>>> col = np.array([0, 0, 1, 2, 2, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> sp.csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
array([[1, 0, 4],
       [0, 0, 5],
       [2, 3, 6]], dtype=int64)

>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> sparse.csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 4],
       [0, 0, 5],
       [2, 3, 6]])
```



## np.identity() 对角阵

```
a = numpy.identity(3,dtype=bool)
print(a)
[[ True False False]
[False True False]
[False False True]]
```



## np.genfromtxt

```python
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
```



## 将矩阵变为对称邻接矩阵

```
# build symmetric adjacency matrix 将矩阵变为对称的邻接矩阵：计算转置，将有向图转为无向图
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
```