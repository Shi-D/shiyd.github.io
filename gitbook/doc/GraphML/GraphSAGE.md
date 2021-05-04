[TOC]

# GraphSAGE

GCN 是一种在图中结合拓扑结构和顶点属性信息学习顶点的 embedding 表示的方法。然而 GCN 要求在一个确定的图中去学习顶点的 embedding，无法直接泛化到在训练过程没有出现过的顶点，即属于一种直推式 (transductive) 的学习。

GraphSAGE 则是一种能够利用顶点的属性信息高效产生未知顶点 embedding 的一种归纳式 (inductive) 学习的框架。

其核心思想是通过学习一个对邻居顶点进行聚合表示的函数来产生目标顶点的embedding向量。



## GraphSAGE 步骤

<img src="https://pic2.zhimg.com/80/v2-f8301d7397b1c703454e5adedbc9d621_1440w.jpg" alt="img" style="zoom:50%;" />

GraphSAGE 的运行流程如上图所示，可以分为三个步骤：

1. 对图中每个顶点邻居顶点进行**采样**

2. 根据**聚合**函数聚合邻居顶点蕴含的信息

3. 得到图中各顶点的向量表示供**下游**任务使用

   

## 采样

对每个顶点采样**一定数量**（确定采样大小）的邻居顶点作为待聚合信息的顶点。设采样数量为k，若顶点邻居数少于 k，则采用**有放回**（即重复采样）的抽样方法，直到采样出 k 个顶点；若顶点邻居数大于 k，则采用无放回的抽样。

当然，若不考虑计算效率，我们完全可以对每个顶点利用其所有的邻居顶点进行信息聚合，这样是信息**无损**的。



## 生成向量的伪代码

<img src="https://pic2.zhimg.com/80/v2-5ac927cd1fca0c700e18e3fc5ef55b45_1440w.jpg" alt="img" style="zoom:50%;" />

这里 K 是网络的层数，也代表着每个顶点能够聚合的邻接点的跳数，如 K=2 的时候每个顶点可以最多根据其 2 跳邻接点的信息学习其自身的 embedding 表示。

在每一层的循环 k 中，对每个顶点v，首先使用 v 的邻接点的 k-1 层的 embedding 表示 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bk-1%7D_u) 来产生其邻居顶点的第 k 层聚合表示 ![[公式]](https://www.zhihu.com/equation?tex=h%5Ek_%7BN%28v%29%7D) ，之后将 ![[公式]](https://www.zhihu.com/equation?tex=h%5Ek_%7BN%28v%29%7D) 和顶点 v 的第 k-1 层表示 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bk-1%7D_v) 进行拼接，经过一个非线性变换产生顶点 v 的第 k 层 embedding 表示 ![[公式]](https://www.zhihu.com/equation?tex=h%5Ek_v) 。



## 聚合函数的选取

由于在图中顶点的邻居是天然无序的，所以我们希望构造出的聚合函数是对称的（即改变输入的顺序，函数的输出结果不变），同时具有较高的表达能力。

#### MEAN aggregator

![[公式]](https://www.zhihu.com/equation?tex=h%5Ek_v%5Cleftarrow+%5Csigma%28%5Cbm%7BW%7D%5Ccdot+%5Ctext%7BMEAN%7D%28%5Cleft+%5C%7B+h_v%5E%7Bk-1%7D+%5Cright+%5C%7D+%5Ccup+%5Cleft%5C%7B+h_u%5E%7Bk-1%7D%2C%5Cforall+u%5Cin+N%28v%29+%5Cright%5C%7D%29)

上式对应于伪代码中的第 4-5 行，直接产生顶点的向量表示，而不是邻居顶点的向量表示。Mean aggregator 将目标顶点和邻居顶点的第 k-1 层向量拼接起来，然后对向量的每个维度进行求均值的操作，将得到的结果做一次非线性变换产生目标顶点的第 k 层表示向量。

#### Pooling aggregator

![[公式]](https://www.zhihu.com/equation?tex=AGGREGATE%5E%7Bpool%7Dk%3D%5Cmax%28%5Cleft%5C%7B%5Csigma%28%5Cbm%7BW%7D%7Bpool%7Dh%5Ek_%7Bu_i%7D%2Bb%29%2C%5Cforall+u_i+%5Cin+N%28v%29%5Cright%5C%7D%29)

Pooling aggregator 先对目标顶点的邻接点表示向量进行一次非线性变换，之后进行一次 pooling 操作 (maxpooling or meanpooling)，将得到结果与目标顶点的表示向量拼接，最后再经过一次非线性变换得到目标顶点的第 k 层表示向量。

#### LSTM aggregator

LSTM 相比简单的求平均操作具有更强的表达能力，然而由于 LSTM 函数不是关于输入对称的，所以在使用时需要对顶点的邻居进行一次**乱序操作**。







> 学习参考：https://zhuanlan.zhihu.com/p/79637787