[TOC]

# Node Classification

借助节点之间的连接关系和少部分已知label的节点，给图中未知label的节点分配label。

单看这个定义，容易和有**监督模型**混淆，如果用有监督模型做节点分类，可以把已知label的节点划为训练集，未知label的节点划为测试集，用训练集训练模型，对测试集做预测。本文所讲的节点分类算法是**集体分类模型（Collective Classification）**，它不显示区分训练集和测试集，它是**利用已知的label和节点之间的连接关系，不断迭代更新未知节点的label，直到整个网络所有节点的label趋于稳定**，即，达到收敛状态。

目前有三种算法，Relational Classification、Iterative Classificaition、Belief Propagation。

这三类算法都基于相同的假设：**节点 $i$ 的label取决于节点 $i$ 的邻居们的label**。

三类算法的计算过程也相同：

> （1）初始化每个节点的label；
>
> （2）设定相连节点之间的相互作用；
>
> （3）根据连接关系在全网多次传播信息，直到全网达到收敛状态。



## Relational Classification

Relational Classification类算法利用节点的连接关系直接进行标签传播（label propagation）。

例如，我们已知9个节点的网络结构如下图，其中四个节点的标签是已知的，我们的目标是预测其余节点的标签。

 <img src="https://images4.pianshen.com/930/64/648853f59ddc2c8338e6412880a4c202.png" alt="img" style="zoom:50%;" />

按照上面介绍的三个步骤来做:

1. 初始化每个未知节点的标签分布为等概率分布，即$P(Y=1)=0.5$ 

​      <img src="https://img-blog.csdnimg.cn/20200601221113627.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI1MjY0MzY=,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 43%;" />

2. 设定每个节点的标签与邻居标签之间的关系如下：

$$
P(Y_i=c)=\frac{1}{\sum_{(i,j) \in E}W(i,j)}\sum_{(i,j)\in E}W(i,j)P(Y_j=c)
$$

​		$W(i, j)$表示节点 $i、j$ 间的边权重。

3. 按照步骤2中设定的作用关系，迭代更新每个节点的label概率分布，直到各未知label的节点的label分布几乎不变，或达到最大迭代次数。需要留心的是，这里迭代采用的是异步方式（asynchronous），即，**每个节点依次更新，且每个节点会利用之前所有节点更新过的label来更新自己的label**，相比于同步方式，异步更新的收敛速度更快，但是会引入不确定性——节点的更新顺序可能影响最终的收敛结果，在实际处理时，可以通过**打乱节点的更新顺序**来降低节点顺序带来的影响。

   

## Iterative Classification

Relational Classification只利用了节点的连接关系和label，忽视了节点的属性信息，Iterative Classification是综合利用节点属性、连接关系和label来迭代更新，最终得到整个网络的label。

这里采用的Iterative Classification的，各节点label是以概率方式给出的，叫做fairness，每个节点都有自己的fairness，一个用户的fairness越低，越可能是欺诈用户。此外，还有另外两个度量指标，分别是边的reliability和商品的goodness，reliability是描述每个评分的可信度，goodness是描述每个商品的好评程度。此外，作者给出了三个指标之间的定量关系，即，如何用某两个指标来计算第三个指标。

然后我们可以借鉴EM算法的思路，固定所有节点的某两个指标，迭代更新所有节点的第三个指标，每一轮迭代依次更新所有节点的三个指标，直到节点的fairness指标保持稳定为止。



## Belief Propagation

Belief Propagation（信念传播）是与条件随机场、贝叶斯网络相关的模型。

关于贝叶斯网络先看纸质的 `理论基础知识` 笔记本进行回顾。



参考：https://blog.csdn.net/u012526436/article/details/106483534



## 实验

### 实验内容

 <img src="https://shiy-d.github.io/ImageHost/hw_node_classification.png" alt="hw_node_classification" style="zoom: 40%;" />

### 实验结果

 <img src="https://shiy-d.github.io/ImageHost/node2.png" alt="node2" style="zoom:36%;" />

### 代码

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

'''构建图'''
nodes = [1,2,4,6,7,9,]
edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 6), (4, 7), (4, 8), (5, 6), (5, 9), (5, 8), (6, 9), (6, 10), (7, 8), (8, 9), (9, 10)]
label = [0.5 for i in range(11)]
label[0] = -1
label[3] = 1
label[5] = 1
label[8] = 0
label[10] = 0
print(nodes)

G = nx.Graph()
G.add_edges_from(edges)

flag = True
round = 0
while flag:
    flag = False
    random.shuffle(nodes)
    for node in nodes:
        # print('node', node)
        weight = 0
        temp_label_node = 0
        for nb in G.neighbors(node):
            weight += 1
            temp_label_node += label[nb]
        # print('weight', weight, 'temp', temp_label_node)
        temp_label_node = temp_label_node/weight
        if label[node] != temp_label_node:
            label[node] = temp_label_node
            flag = True
            # print('flag', flag)
    # print('flag', flag)
    print('round',round, ':')
    for j in range(1, 11):
        print(str(j)+':'+str(label[j]), end=' ')
    print()
    round += 1

com = list()
nodes_data = G.nodes.data()
for node in nodes_data:
    label[node[0]] = 1 if label[node[0]]>0.5 else 0
    com.append(label[node[0]])

print('com', com)

cmap = plt.cm.get_cmap('viridis', 12)
print(cmap)
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, width=0.2, cmap=cmap, node_color=com)
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.show()


```

