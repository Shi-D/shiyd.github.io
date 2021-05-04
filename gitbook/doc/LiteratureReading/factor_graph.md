[TOC]

# 因子图

Factor Graph 是概率图的一种，概率图有很多种，最常见的就是Bayesian Network (贝叶斯网络)和Markov Random Fields(马尔可夫随机场)。在概率图中，求某个变量的边缘分布是常见的问题。这问题有很多求解方法，其中之一就是可以把Bayesian Network和Markov Random Fields 转换成Factor Graph，然后用sum-product算法求解。基于Factor Graph可以用sum-product算法可以高效的求各个变量的边缘分布。

sum-product算法，也叫belief propagation，有两种消息：

* 一种是变量(Variable)到函数(Function)的消息(就是方块到圆的消息)，m：x→f 。
* 一种是函数(Function)到变量(Variable)的消息(就是圆到方块的消息)，m：f→x。

## 1 因子分解

设 ![[公式]](https://www.zhihu.com/equation?tex=X+%3D+%5Cleft%5C%7B+x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D+%5Cright%5C%7D) 是一个变量集合，对于每个 ![[公式]](https://www.zhihu.com/equation?tex=i) ， ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D%5Cin+A_%7Bi%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D%29) 为一关于 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D) 的实值函数，即函数定义域为 ![[公式]](https://www.zhihu.com/equation?tex=S%3DA_%7B1%7D%5Ctimes+A_%7B1%7D%5Ctimes+A_%7B2%7D%5Ctimes+...%5Ctimes++A_%7Bn%7D) ,值域为 ![[公式]](https://www.zhihu.com/equation?tex=R+) 。

若 ![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D%29) 可以分解为几个局部函数的乘积，每个函数的参数均为 ![[公式]](https://www.zhihu.com/equation?tex=X+) 的子集，即

![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D%29+%3D+%5Cprod_%7Bj+%5Cin+J%7D%5E%7B%7Df_%7Bj%7D%28X_%7Bj%7D%29+%5Ctag1)

其中， ![[公式]](https://www.zhihu.com/equation?tex=J) 是一个离散的索引集合， <img src="https://www.zhihu.com/equation?tex=X_%7Bj%7D" alt="[公式]" style="zoom:94%;" /> 是 ![[公式]](https://www.zhihu.com/equation?tex=X+) 的一个子集， ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bj%7D%28X_%7Bj%7D%29) 是以 <img src="https://www.zhihu.com/equation?tex=X_%7Bj%7D" alt="[公式]" style="zoom:94%;" /> 中元素为参数的函数。

则称公式（1）为 ![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D%29) 的**一个**因子分解，其中 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bj%7D%28X_%7Bj%7D%29) 为 ![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D%29) 的一个因子。



## 2 边缘函数

对第1部分提到的函数 ![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D%29) ，对应了 ![[公式]](https://www.zhihu.com/equation?tex=n) 个函数 ![[公式]](https://www.zhihu.com/equation?tex=p_%7Bi%7D%28x_%7Bi%7D%29) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=p_%7Bi%7D%28x_%7Bi%7D%29) 满足

![[公式]](https://www.zhihu.com/equation?tex=p_%7Bi%7D%28x_%7Bi%7D%29%3D%5Csum_%7BX-x_%7Bi%7D%7D++%7Bp%28x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D%29%7D+%5Ctag2)

其中， ![[公式]](https://www.zhihu.com/equation?tex=X-x_%7Bi%7D) 为 ![[公式]](https://www.zhihu.com/equation?tex=X) 中除 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D) 之外的其他元素组成的集合，则称![[公式]](https://www.zhihu.com/equation?tex=p_%7Bi%7D%28x_%7Bi%7D%29)为 ![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D%29) 的一个边缘函数 。



## 3 Sum-Product算法

因子图是一个能够表示公式（1）所示的因子分解结构的二部图，图中节点包括表示了每个变量 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D) 的变量节点及表示每个局部函数 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bj%7D) 的因子节点，图中边表示变量![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D)为 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bj%7D) 的一个参数。

设 ![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2Cx_%7B3%7D%2Cx_%7B4%7D%2Cx_%7B5%7D%29) 为包含5个变量的函数，若其可分解为

![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2Cx_%7B3%7D%2Cx_%7B4%7D%2Cx_%7B5%7D%29+%3D+f_%7BA%7D%28x_%7B1%7D%29f_%7BB%7D%28x_%7B2%7D%29f_%7BC%7D%28x_%7B1%7D%2Cx_%7B2%7D%2Cx_%7B3%7D%29f_%7BD%7D%28x_%7B3%7D%2Cx_%7B4%7D%29f_%7BE%7D%28x_%7B3%7D%2Cx_%7B5%7D%29%5Ctag3)

则公式（3）所示的因子分解可以有图1表达。

<img src="https://pic2.zhimg.com/80/v2-feb7479e8e8a0b3ea59b69f2c7ac5439_1440w.jpg" alt="img" style="zoom:50%;" />

因子图主要有两个用途，1）表达因子分解的结构，2）计算边缘函数。

## 4 Sum-Product算法

Sum-Product 算法主要被用来计算边缘分布（边缘函数），因子图可以看做是一个树结构。若计算 ![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B1%7D%2Cx_%7B2%7D%2Cx_%7B3%7D%2Cx_%7B4%7D%2Cx_%7B5%7D%29) 中的边缘函数 ![[公式]](https://www.zhihu.com/equation?tex=p%28x_%7B3%7D%29) ,则将图1可看做以变量节点 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B3%7D) 为根节点的树，如下图所示。

<img src="https://pic2.zhimg.com/80/v2-264865fe064d0192364536a466188a75_1440w.jpg" alt="img" style="zoom:67%;" />

因子图的树结构表示，我们可以将联合概率分布的因子分解成若干组，每一组对应了根节点的相邻的因子节点的结合。

为了方便描述，设 ![[公式]](https://www.zhihu.com/equation?tex=X%3D%5Cleft%5C%7B+x%2Cx_%7B1%7D%2C...%2Cx_%7Bn%7D+%5Cright%5C%7D) , ![[公式]](https://www.zhihu.com/equation?tex=X-x%3D%5Cleft+%5C%7Bx_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bn%7D+%5Cright%5C%7D%5C)，则

![[公式]](https://www.zhihu.com/equation?tex=p%28x%2Cx_%7B1%7D%2C...%2Cx_%7Bn%7D%29+%3D+%5Cprod_%7Bs+%5Cin+ng%28x%29%7D%5E%7B%7DF_%7Bs%7D%28x%2CX_%7Bs%7D%29+%5Ctag4)

其中 ![[公式]](https://www.zhihu.com/equation?tex=x) 表示因子图树结构的根节点， ![[公式]](https://www.zhihu.com/equation?tex=ng%28x%29) 表示与 ![[公式]](https://www.zhihu.com/equation?tex=x) 相邻的因子节点集合， ![[公式]](https://www.zhihu.com/equation?tex=s) 表示 ![[公式]](https://www.zhihu.com/equation?tex=ng%28x%29) 中任一元素， ![[公式]](https://www.zhihu.com/equation?tex=X_%7Bs%7D) 表示通过因子节点 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bs%7D) 与 ![[公式]](https://www.zhihu.com/equation?tex=x) 连接的所有变量的集合， ![[公式]](https://www.zhihu.com/equation?tex=F_%7Bs%7D%28x%2CX_%7Bs%7D%29) 表示该分组中与因子 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bs%7D) 相关联的所有因子的乘积。

为了求边缘分布，将公式（4）代入公式（2）可得边缘分布如下：

![[公式]](https://www.zhihu.com/equation?tex=p%28x%29%3D%5Csum_%7BX-x%7D++%7B%5Cprod_%7Bs+%5Cin+ng%28x%29%7D%5E%7B%7DF_%7Bs%7D%28x%2CX_%7Bs%7D%29+%7D+%5Ctag5)

设 ![[公式]](https://www.zhihu.com/equation?tex=S%3D%5Cleft%7C+ng%28x%29%5Cright%7C) , 由树结构可知，公式（5）中的![[公式]](https://www.zhihu.com/equation?tex=X_%7Bs%7D)互不相交，即 ![[公式]](https://www.zhihu.com/equation?tex=X_%7Bi%7D%5Ccap+X_%7Bj%7D%3D%5CPhi) ，可设 ![[公式]](https://www.zhihu.com/equation?tex=X-x%3DX_%7B1%7D%5Ccup+X_%7B2%7D+%5Ccup+...+%5Ccup+X_%7BS%7D+) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=X_%7Bs%7D%3D%5Cleft%5C%7B+x_%7Bs1%7D%2Cx_%7Bs2%7D%2C...%2Cx_%7BsM%7D+%5Cright%5C%7D%7B%7D) ，且![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7BX-x%7D%3D%5Csum_%7BX_%7B1%7D%7D...%5Csum_%7BX_%7BS%7D%7D) ，则公式（5）可化为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+p%28x%29+%26+%3D+%5Csum_%7BX_%7B1%7D%7D...%5Csum_%7BX_%7BS%7D%7D++%7B%5Cprod_%7Bs+%5Cin+ng%28x%29%7D%5E%7B%7DF_%7Bs%7D%28x%2CX_%7Bs%7D%29+%7D+%5C%5C++%26+%3D++%5Csum_%7BX_%7B1%7D%7D...%5Csum_%7BX_%7BS-1%7D%7D%28%5Csum_%7BX_%7BS%7D%7D++%7B%5Cprod_%7Bs+%5Cin+ng%28x%29%7D%5E%7B%7DF_%7Bs%7D%28x%2CX_%7Bs%7D%29+%7D+%29%5C%5C+%5Cend%7Baligned%7D%5Ctag6)

对公式（6）括号中中提取公因式得，

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+p%28x%29++%26+%3D++%5Csum_%7BX_%7B1%7D%7D...%5Csum_%7BX_%7BS-1%7D%7D%7B%28%5Cprod_%7Bs+%5Cin+ng%28x%29-S%7D%5E%7B%7DF_%7Bs%7D%28x%2CX_%7Bs%7D%29%29%28%5Csum_%7BX_%7BS%7D%7D+F_%7BS%7D%28x%2CX_%7BS%7D%29+%7D%29+%5C%5C+%26+%3D++%5Csum_%7BX_%7B1%7D%7D+F_%7B1%7D%28x%2CX_%7B1%7D%29%5Csum_%7BX_%7B2%7D%7D+F_%7B2%7D%28x%2CX_%7B2%7D%29...%5Csum_%7BX_%7BS%7D%7D+F_%7BS%7D%28x%2CX_%7BS%7D%29+%5C%5C+%26+%3D++%5Cprod_%7Bs+%5Cin+ng%28x%29%7D%5E%7B%7D%5Csum_%7BX_%7Bs%7D%7D+F_%7Bs%7D%28x%2CX_%7Bs%7D%29%5C%5C+%5Cend%7Baligned%7D%5Ctag7)

记，

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D%5Cmu_%7Bf_%7Bs%7D%5Crightarrow+x%7D%28x%29+%26%3D%5Csum_%7BX_%7Bs%7D%7D+F_%7Bs%7D%28x%2CX_%7Bs%7D%29+%5C%5C+%5Cend%7Baligned%7D+%5Ctag8)

![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_%7Bf_%7Bs%7D%5Crightarrow+x%7D%28x%29) 可以看做从因子结点 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bs%7D) 到变量结点 ![[公式]](https://www.zhihu.com/equation?tex=x) 的信息（message），表示与变量节点 ![[公式]](https://www.zhihu.com/equation?tex=x) 相连的第 ![[公式]](https://www.zhihu.com/equation?tex=s) 个因子节点传给 ![[公式]](https://www.zhihu.com/equation?tex=x) 的结果，其自变量为 ![[公式]](https://www.zhihu.com/equation?tex=x) ，则边缘分布 ![[公式]](https://www.zhihu.com/equation?tex=p%28x%29) 可化为，

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+p%28x%29++%26+%3D++%5Cprod_%7Bs+%5Cin+ng%28x%29%7D%5E%7B%7D%5Cmu_%7Bf_%7Bs%7D%5Crightarrow+x%7D%28x%29%5C%5C+%5Cend%7Baligned%7D%5Ctag9)

求解过程如下图所示。

![img](https://pic1.zhimg.com/80/v2-660c4a6141681431d5b1b94e513091a0_1440w.jpg)

从上图中可以看出，因子节点 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bs%7D) 与集合 ![[公式]](https://www.zhihu.com/equation?tex=X_%7Bs%7D%3D%5Cleft%5C%7B+x_%7Bs1%7D%2Cx_%7Bs2%7D%2C...%2Cx_%7BsM%7D+%5Cright%5C%7D%7B%7D) 中的变量连接，若干变量节点连接，由上图可知，因子节点 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bs%7D) 会收到的 ![[公式]](https://www.zhihu.com/equation?tex=X_%7Bs%7D) 中变量节点发来的信息，![[公式]](https://www.zhihu.com/equation?tex=f_%7Bs%7D)收到所有节点发来的信息之后，经过某种运算，得出一个新结果，这个新结果即为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_%7Bf_%7Bs%7D%5Crightarrow+x%7D%28x%29) 。

![[公式]](https://www.zhihu.com/equation?tex=f_%7Bs%7D) 执行的运算包括两步，第一步为将收到的运算与因子自身做乘法(积运算)，其结果为关于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+x%2Cx_%7Bs1%7D%2Cx_%7Bs2%7D%2C...%2Cx_%7BsM%7D+%5Cright%5C%7D%7B%7D) 的函数，即公式（8）中的 ![[公式]](https://www.zhihu.com/equation?tex=F_%7Bs%7D%28x%2CX_%7Bs%7D%29+) ，第二步为对 ![[公式]](https://www.zhihu.com/equation?tex=F_%7Bs%7D%28x%2CX_%7Bs%7D%29+) 求关于 ![[公式]](https://www.zhihu.com/equation?tex=X_%7Bs%7D) 中所有元素的边缘概率（和运算）。设变量节点 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bsi%7D) 发来的信息结果为 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bs%7D%28x_%7Bsi%7D%2CF_%7Bsi%7D%29) ,其中 ![[公式]](https://www.zhihu.com/equation?tex=F_%7Bsi%7D) 表示与变量节点 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bsi%7D) 相邻的除 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bs%7D) 之外的因子节点集合，则第一步得出的结果为

![[公式]](https://www.zhihu.com/equation?tex=F_%7Bs%7D%28x%2CX_%7Bs%7D%29+%3Df_s%28x%2CX_%7Bs%7D%29G_%7Bs1%7D%28x_%7Bs1%7D%2CF_%7Bs1%7D%29...G_%7BsM%7D%28x_%7BsM%7D%2CF_%7BsM%7D%29%5Ctag%7B10%7D)

变量节点 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bsi%7D) 又与若干因子节点相邻，其发送的信息结果 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bs%7D%28x_%7Bsi%7D%2CF_%7Bsi%7D%29) 其实是以变量节点 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bsi%7D)为根的因子子树表示的联合概率关于根节点的局部边缘概率![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7BP%7D%28x_%7Bsi%7D%29)(相对与因子子树)，可利用（9）递归计算。而 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bsi%7D%28x_%7Bsi%7D%2CF_%7Bsi%7D%29) 为变量节点![[公式]](https://www.zhihu.com/equation?tex=x_%7Bsi%7D)向因子节点 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bs%7D) （![[公式]](https://www.zhihu.com/equation?tex=x_%7Bsi%7D%5Crightarrow+f_%7Bs%7D)）传递的信息结果，可记其为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_%7Bx_%7Bsi%7D%5Crightarrow+f_%7Bs%7D%7D%28x_%7Bsi%7D%29) ，其表示以变量节点 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bsi%7D)为根的因子子树表示的联合概率关于根节点的边缘概率 ![[公式]](https://www.zhihu.com/equation?tex=P%28x_%7Bsi%7D%29) 。则公式（10）可表示为

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+F_%7Bs%7D%28x%2CX_%7Bs%7D%29+%26%3Df_s%28x%2CX_%7Bs%7D%29%5Cprod_%7Bm%3D1%7D%5E%7BM%7DG_%7Bsm%7D%28x_%7Bsm%7D%2CF_%7Bsm%7D%29+%5C%5C+%26%3Df_s%28x%2CX_%7Bs%7D%29%5Cprod_%7Bm%3D1%7D%5E%7BM%7D%5Cmu_%7Bx_%7Bsm%7D%5Crightarrow+f_%7Bs%7D%7D%28x_%7Bsm%7D%29%5Cend%7Baligned%7D%5Ctag%7B11%7D)

公式（8）可表示为

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D%5Cmu_%7Bf_%7Bs%7D%5Crightarrow+x%7D%28x%29+%26%3D%5Csum_%7Bx_%7Bs1%7D%7D+%5Csum_%7Bx_%7Bs2%7D%7D...%5Csum_%7Bx_%7BsM%7D%7Df_s%28x%2CX_%7Bs%7D%29%5Cprod_%7Bm%3D1%7D%5E%7BM%7D%5Cmu_%7Bx_%7Bsm%7D%5Crightarrow+f_%7Bs%7D%7D%28x_%7Bsm%7D%29+%5C%5C++%26%3D%5Csum_%7Bx_%7Bs1%7D%7D+%5Csum_%7Bx_%7Bs2%7D%7D...%5Csum_%7Bx_%7BsM%7D%7Df_s%28x%2CX_%7Bs%7D%29%5Cprod_%7Bm%3D1%7D%5E%7BM%7D%5Ctilde%7BP%7D%28x_%7Bsm%7D%29+%5C%5C%5Cend%7Baligned%7D+%5Ctag%7B12%7D)

因子图可以用来求边缘概率，既可以求单个根节点的边缘概率，也也可以求所有变量节点的边缘概率。



参考：https://zhuanlan.zhihu.com/p/84210564