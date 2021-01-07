# Markov

## 隐马尔可夫模型HMM

https://www.cnblogs.com/skyme/p/4651331.html

**Viterbi算法**（维特比算法）



## 马尔可夫决策过程MDP

https://zhuanlan.zhihu.com/p/28084942

### **马尔科夫过程 Markov Process**

- **马尔科夫性 Markov Property**

某一状态信息包含了所有相关的历史，只要当前状态可知，所有的历史信息都不再需要，当前状态就可以决定未来，则认为该状态具有**马尔科夫性**。

- **马尔科夫过程 Markov Property**

**马尔科夫过程** 又叫马尔科夫链(Markov Chain)，它是一个无记忆的随机过程，可以用一个元组***<S,P>***表示，其中S是有限数量的状态集，P是状态转移概率矩阵。

### **马尔科夫奖励过程 Markov Reward Process**

马尔科夫奖励过程在马尔科夫过程的基础上增加了**奖励R**和**衰减系数γ**：***<S,P,R,γ>***。

**奖励函数**

R是一个奖励函数。S状态下的奖励是某一时刻(t)处在状态s下在下一个时刻(t+1)能获得的奖励期望：![[公式]](https://www.zhihu.com/equation?tex=R_%7Bs%7D+%3D+E%5BR_%7Bt%2B1%7D+%7C+S_%7Bt%7D+%3D+s+%5D)

很多听众纠结为什么**奖励是t+1时刻的**。照此理解起来相当于离开这个状态才能获得奖励而不是进入这个状态即获得奖励。David指出这仅是一个约定，为了在描述RL问题中涉及到的观测O、行为A、和奖励R时比较方便。他同时指出如果把奖励改为 ![[公式]](https://www.zhihu.com/equation?tex=R_t) 而不是 ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bt%2B1%7D) ，只要规定好，本质上意义是相同的，在表述上可以把奖励描述为“当进入某个状态会获得相应的奖励”。

**衰减系数 Discount Factor** 

γ∈ [0, 1]，它的引入有很多理由，其中优达学城的“机器学习-强化学习”课程对其进行了非常有趣的数学解释。David也列举了不少原因来解释为什么引入衰减系数，其中有数学表达的方便，避免陷入无限循环，远期利益具有一定的不确定性，符合人类对于眼前利益的追求，符合金融学上获得的利益能够产生新的利益因而更有价值等等。

**收获 Return**

定义：收获 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%7D) 为在一个马尔科夫奖励链上从t时刻开始往后所有的奖励的有衰减的总和。也有翻译成“收益”或"回报"。公式如下：

![img](https://pic1.zhimg.com/80/v2-e5e691ff4b754db8f893dfd367107600_1440w.png)

其中衰减系数体现了未来的奖励在当前时刻的价值比例，在k+1时刻获得的奖励R在t时刻的体现出的价值是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma%5Ek+R) ，γ接近0，则表明趋向于“近视”性评估；γ接近1则表明偏重考虑远期的利益。

**价值函数 Value Function**

价值函数给出了某一状态或某一行为的长期价值。

定义：一个马尔科夫奖励过程中某一状态的**价值函数**为**从该状态开始**的马尔可夫链收获的期望：

![[公式]](https://www.zhihu.com/equation?tex=v%28s%29+%3D+E+%5B+G_%7Bt%7D+%7C+S_%7Bt%7D+%3D+s+%5D)

注：价值可以仅描述状态，也可以描述某一状态下的某个行为，在一些特殊情况下还可以仅描述某个行为。在整个视频公开课中，除了特别指出，约定用**状态价值函数**或**价值函数**来描述针对状态的价值；用**行为价值函数**来描述某一状态下执行某一行为的价值，严格意义上说行为价值函数是**状态行为对价值函数**的简写。

**举例说明收获和价值的计算**

为方便计算，把“学生马尔科夫奖励过程”示例图表示成下表的形式。表中第二行对应各状态的即时奖励值，蓝色区域数字为状态转移概率，表示为从所在行状态转移到所在列状态的概率：

![img](https://pic3.zhimg.com/80/v2-52c5d21082994b4cc1d4aac0fe4f58ba_1440w.png)

考虑如下4个马尔科夫链。现计算当γ= 1/2时，在t=1时刻（![[公式]](https://www.zhihu.com/equation?tex=S_%7B1%7D+%3D+C_%7B1%7D)）时状态 ![[公式]](https://www.zhihu.com/equation?tex=S_%7B1%7D) 的收获分别为：
![img](https://pic2.zhimg.com/80/v2-91921a745909435f7b984d1dae5ef271_1440w.png)

从上表也可以理解到，收获是针对一个马尔科夫链中的**某一个状态**来说的。

**价值函数的推导**

 

- **Bellman方程 - MRP**

先尝试用价值的定义公式来推导看看能得到什么：

![img](https://pic3.zhimg.com/80/v2-fda247960872e2cb7653bcb89729626a_1440w.png)

这个推导过程相对简单，仅在导出最后一行时，将 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%2B1%7D) 变成了 ![[公式]](https://www.zhihu.com/equation?tex=v%28S_%7Bt%2B1%7D%29) 。其理由是收获的期望等于收获的期望的期望。下式是针对MRP的Bellman方程：

![img](https://pic3.zhimg.com/80/v2-d9bf4d39fba6d6afcb9e0e8cae734242_1440w.png)

通过方程可以看出 ![[公式]](https://www.zhihu.com/equation?tex=v%28s%29) 由两部分组成，一是该状态的即时奖励期望，即时奖励期望等于即时奖励，因为根据即时奖励的定义，它与下一个状态无关；另一个是下一时刻状态的价值期望，可以根据下一时刻状态的概率分布得到其期望。如果用s’表示s状态下一时刻任一可能的状态，那么Bellman方程可以写成：

![img](https://pic1.zhimg.com/80/v2-1164fecb7bf77d8210343e53c4fa7ac8_1440w.png)

- **方程的解释**

下图已经给出了γ=1时各状态的价值（该图没有文字说明γ=1，根据视频讲解和前面图示以及状态方程的要求，γ必须要确定才能计算），状态 ![[公式]](https://www.zhihu.com/equation?tex=C_%7B3%7D) 的价值可以通过状态Pub和Pass的价值以及他们之间的状态转移概率来计算：

![[公式]](https://www.zhihu.com/equation?tex=4.3+%3D+-2+%2B+1.0+%2A+%28+0.6+%2A+10+%2B+0.4+%2A+0.8+%29)



![img](https://pic4.zhimg.com/80/v2-a8997be4d72fcb8faaee4db82db495b3_1440w.png)



- **Bellman方程的矩阵形式和求解**

![img](https://pic4.zhimg.com/80/v2-444fc8bffca56f64f6599818800c54df_1440w.png)

结合矩阵的具体表达形式还是比较好理解的：

![img](https://pic3.zhimg.com/80/v2-071d680f97a7cfc7199f03a700b1f9a2_1440w.png)

Bellman方程是一个线性方程组，因此理论上解可以直接求解：

![img](https://pic4.zhimg.com/80/v2-ee67be43e30fababfab0fb4820db303f_1440w.png)

实际上，计算复杂度是 ![[公式]](https://www.zhihu.com/equation?tex=O%28n%5E%7B3%7D%29) ， ![[公式]](https://www.zhihu.com/equation?tex=n) 是状态数量。因此直接求解仅适用于小规模的MRPs。大规模MRP的求解通常使用迭代法。常用的迭代方法有：动态规划Dynamic Programming、蒙特卡洛评估Monte-Carlo evaluation、时序差分学习Temporal-Difference，后文会逐步讲解这些方法。



### **马尔科夫决策过程 Markov Decision Process**

相较于马尔科夫奖励过程，马尔科夫决定过程多了一个行为集合A，它是这样的一个元组: <S, A, P, R, γ>。看起来很类似马尔科夫奖励过程，但这里的P和R都与具体的**行为**a对应，而不像马尔科夫奖励过程那样仅对应于某个**状态**，A表示的是有限的行为的集合。具体的数学表达式如下：


 ![img](https://pic2.zhimg.com/80/v2-8d5223ece5e1c82928b164a7a7e589e9_1440w.png)

 ![img](https://pic4.zhimg.com/80/v2-0e748583bfa697a166935c91226fba6f_1440w.png)

