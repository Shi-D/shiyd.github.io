# Struc2Vec

DeepWalk,LINE,Node2Vec,SDNE 这几个 graph embedding 方法。这些方法都是基于近邻相似的假设的。其中 DeepWalk,Node2Vec 通过随机游走在图中采样顶点序列来构造顶点的近邻集合。LINE 显式的构造邻接点对和顶点的距离为 1 的近邻集合。SDNE 使用邻接矩阵描述顶点的近邻结构。

事实上，在一些场景中，两个不是近邻的顶点也可能拥有很高的相似性，对于这类相似性，上述方法是无法捕捉到的。Struc2Vec 就是针对这类场景提出的。Struc2Vec 的论文发表在2017年的KDD会议中。

## 算法原理

### 相似度定义

Struc2Vec是从空间结构相似性的角度定义顶点相似度的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019021411110263.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTIxNTEyODM=,size_16,color_FFFFFF,t_70)

直观来看，具有相同度数的顶点是结构相似的，若各自邻接顶点仍然具有相同度数，那么他们的相似度就更高。

### 顶点对距离定义

令 $R_k(u)$ 表示到顶点u距离为k的顶点集合，则 $R_1(u)$ 表示是 u 的直接相连近邻集合。
令 $s(S)$ 表示顶点集合S SS的有序度序列。



