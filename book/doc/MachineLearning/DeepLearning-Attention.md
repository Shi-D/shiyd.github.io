[TOC]

# Attention

## 一、简介

投入更多的注意力到某些重点的地方，而忽略那些不重要的部分。

## 二、作用目标

#### 通道

关注这个东西是什么

打分函数维度不在乎空间的维度，而是channel的数量

对当前层的每一张特征图形打分

#### 空间

关注在哪

在乎空间信息，而不在乎channel的信息

对当前层所有特征图的每一个位置进行打分

## 三、相关Attention

**1、SENet**

**2、Non-local**

**3、SKNet**

**4、BiseNet**

...

## 四、Encoder-Decoder框架

如下图，是文本处理领域里常用的Encoder-Decoder框架最抽象的一种表示。

![encoder-decoder](https://pic1.zhimg.com/80/v2-a5093fc7c0c4942b1d47e7cd2e65ea3b_1440w.jpg?source=1940ef5c)

以NLP为例，输入source，要求输出Target，

![encoder-decoder](https://pic2.zhimg.com/80/v2-4ab3a2d834a45581c32ea62aec28e428_1440w.jpg?source=1940ef5c)

Encoder顾名思义就是对输入句子Source进行编码，将输入句子通过非线性变换转化为中间语义表示C，

![encoder-decoder](https://pic1.zhimg.com/80/v2-749be2e4a08d8e8b3874d23b0f6c3f22_1440w.jpg?source=1940ef5c)

对于解码器Decoder来说，其任务是根据句子Source的中间语义表示C和之前已经生成的历史信息，

![img](https://pic4.zhimg.com/50/v2-18682b9de059b0ea2d1a25a45bea708a_hd.jpg?source=1940ef5c)

![img](https://pic1.zhimg.com/80/v2-0036edc4b2888e0df612585dd5f28ecd_1440w.jpg?source=1940ef5c)

每个y_i都依次这么产生，那么看起来就是整个系统根据输入句子Source生成了目标句子Target。如果Source是中文句子，Target是英文句子，那么这就是解决机器翻译问题的Encoder-Decoder框架；如果Source是一篇文章，Target是概括性的几句描述语句，那么这是文本摘要的Encoder-Decoder框架；如果Source是一句问句，Target是一句回答，那么这是问答系统或者对话机器人的Encoder-Decoder框架。由此可见，在文本处理领域，Encoder-Decoder的应用领域相当广泛。

## 五、Soft Attention

**下面以机器翻译为例**

**分心模型**

上面展示的Encoder-Decoder框架是没有体现出“注意力模型”的，即注意力不集中的分心模型。

![img](https://pic1.zhimg.com/80/v2-68a9bfb44a4c7df485204727cdc8bdf2_1440w.jpg?source=1940ef5c)

其中f是Decoder的非线性变换函数。从这里可以看出，在生成目标句子的单词时，不论生成哪个单词，它们使用的输入句子Source的语义编码C都是一样的，没有任何区别。

而语义编码C是由句子Source的每个单词经过Encoder 编码产生的，意味着句子Source中任意单词对生成某个目标单词yi来说影响力都是相同的，所以说这个模型没有体现出注意力的缘由。

**引入注意力机制的原因：**

没有引入注意力的模型在输入句子比较短的时候问题不大，但是如果输入句子比较长，此时所有语义完全通过一个中间语义向量来表示，单词自身的信息已经消失，可想而知会丢失很多细节信息，这也是为何要引入注意力模型的重要原因。

因此目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。这意味着在生成每个单词yi的时候，原先都是相同的中间语义表示C会被替换成**根据当前生成单词而不断变化的C_i**。

![img](https://pic2.zhimg.com/80/v2-92302aa42ae10c63627663430ab60f73_1440w.jpg?source=1940ef5c)

即生成目标句子单词的过程成了下面的形式：



![img](https://pic1.zhimg.com/80/v2-4b7a63328249254920b9ef058ec6fe86_1440w.jpg?source=1940ef5c)



而每个Ci可能对应着不同的源语句子单词的注意力分配概率分布，比如对于上面的英汉翻译来说，其对应的信息可能如下：



![img](https://pic1.zhimg.com/50/v2-69bf09b870b5472cfc5bf24c892bf157_hd.jpg?source=1940ef5c)

其中，f2函数代表Encoder对输入英文单词的某种变换函数，比如如果Encoder是用的RNN模型的话，这个f2函数的结果往往是某个时刻输入xi后隐层节点的状态值；g代表Encoder根据单词的中间表示合成整个句子中间语义表示的变换函数，一般的做法中，g函数就是对构成元素加权求和，即下列公式：

![img](https://pic4.zhimg.com/80/v2-a204b9a5817be9a7fbc0a1abfa6b9ab2_1440w.jpg?source=1940ef5c)

其中，Lx代表输入句子Source的长度，aij代表在Target输出第i个单词时Source输入句子中第j个单词的注意力分配系数，而hj则是Source输入句子中第j个单词的语义编码。假设下标i就是上面例子所说的“ 汤姆” ，那么Lx就是3，h1=f(“Tom”)，h2=f(“Chase”),h3=f(“Jerry”)分别是输入句子每个单词的语义编码，对应的注意力模型权值则分别是0.6,0.2,0.2，所以g函数本质上就是个加权求和函数。如果形象表示的话，翻译中文单词“汤姆”的时候，数学公式对应的中间语义表示Ci的形成过程类似图4。

![img](https://pic1.zhimg.com/80/v2-b89c84193f325482e145911b590faa93_1440w.jpg?source=1940ef5c)



这时，如何计算图中的al1、al2...呢？即求输入句子单词注意力分配概率分布值？

对于采用RNN的Decoder来说，在时刻i，如果要生成yi单词，我们是可以知道Target在生成Yi之前的时刻i-1时，隐层节点i-1时刻的输出值Hi-1的，而我们的目的是要计算生成Yi时输入句子中的单词“Tom”、“Chase”、“Jerry”对Yi来说的注意力分配概率分布，那么可以用Target输出句子i-1时刻的隐层节点状态Hi-1去一一和输入句子Source中每个单词对应的RNN隐层节点状态hj进行对比，即通过函数F(hj,Hi-1)来获得目标单词yi和每个输入单词对应的对齐可能性，这个F函数在不同论文里可能会采取不同的方法，然后函数F的输出经过Softmax进行归一化就得到了符合概率分布取值区间的注意力分配概率分布数值。

![img](https://pic2.zhimg.com/80/v2-ac1c016e17a1a681a1dd7da0fb18d1e8_1440w.jpg?source=1940ef5c)

## 六、Attention机制的本质思想

![img](https://pic2.zhimg.com/50/v2-24927f5c33083c1322bc16fa9feb38fd_hd.jpg?source=1940ef5c)

我们可以这样来看待Attention机制：将Source中的构成元素想象成是由一系列的**<Key,Value>**数据对构成，此时给定**Target**中的某个元素**Query**，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以**本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数**。即可以将其本质思想改写为如下公式：

![img](https://pic4.zhimg.com/50/v2-76cac5c196e43afc8338712b6a41d491_hd.jpg?source=1940ef5c)

至于Attention机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程：**第一个过程是根据Query和Key计算权重系数**，**第二个过程根据权重系数对Value进行加权求和**。而第一个过程又可以细分为两个阶段：**第一个阶段根据Query和Key计算两者的相似性或者相关性**；**第二个阶段对第一阶段的原始分值进行归一化处理**；这样，可以将Attention的计算过程抽象为如图10展示的三个阶段。

![img](https://pic2.zhimg.com/50/v2-07c4c02a9bdecb23d9664992f142eaa5_hd.jpg?source=1940ef5c)

































参考：https://www.zhihu.com/question/68482809/answer/264632289