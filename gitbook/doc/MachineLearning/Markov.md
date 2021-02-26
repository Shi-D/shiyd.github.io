# Markov

## 隐马尔可夫模型HMM

https://www.cnblogs.com/skyme/p/4651331.html

**Viterbi算法**（维特比算法）



## 马尔可夫决策过程MDP

https://leovan.me/cn/2020/05/markov-decision-process/

### **马尔可夫过程 Markov Process**

- **马尔可夫性 Markov Property**

某一状态信息包含了所有相关的历史，只要当前状态可知，所有的历史信息都不再需要，当前状态就可以决定未来，则认为该状态具有**马尔可夫性**。

- **马尔可夫过程 Markov Property**

**马尔可夫过程** 又叫**马尔可夫链(Markov Chain)**，它是一个无记忆的随机过程，可以用一个元组***<S,P>***表示，其中S是有限数量的状态集，P是状态转移概率矩阵。

### **马尔可夫奖励过程MRP Markov Reward Process**

马尔可夫奖励过程在马尔可夫过程的基础上增加了**奖励R**和**衰减系数γ**：***<S,P,R,γ>***。

* **奖励函数**

$\mathcal{R}$为收益函数，$\mathcal{R}_s = \mathbb{E} \left[R_t | S_{t-1} = s\right]$

R是一个奖励函数。S状态下的奖励是某一时刻(t)处在状态s下在下一个时刻(t+1)能获得的奖励期望：![[公式]](https://www.zhihu.com/equation?tex=R_%7Bs%7D+%3D+E%5BR_%7Bt%2B1%7D+%7C+S_%7Bt%7D+%3D+s+%5D)

* **衰减系数 Discount Factor** 

$\gamma \in \left[0, 1\right]$为**衰减系数**，也叫**折扣率**。

* **收获(期望回报) Return**

定义：收获 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%7D) 为在一个马尔可夫奖励链上从t时刻开始往后所有的奖励的有衰减的总和。公式如下：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$
其中衰减系数体现了未来的奖励在当前时刻的价值比例，在k+1时刻获得的奖励R在t时刻的体现出的价值是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma%5Ek+R) ，γ接近0，则表明趋向于“近视”性评估；γ接近1则表明偏重考虑远期的利益。

* **价值函数 Value Function**

价值函数给出了某一状态或行为的长期价值。

定义：一个马尔可夫奖励过程中某一状态的**价值函数**为**从该状态开始**的马尔可夫链收获的期望：

![[公式]](https://www.zhihu.com/equation?tex=v%28s%29+%3D+E+%5B+G_%7Bt%7D+%7C+S_%7Bt%7D+%3D+s+%5D)
$$
\begin{aligned}
v(s) &=\mathbb{E}\left[G_{t} | S_{t}=s\right] \\
&=\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots | S_{t}=s\right] \\
&=\mathbb{E}\left[R_{t+1}+\gamma\left(R_{t+2}+\gamma R_{t+3}+\ldots\right) | S_{t}=s\right] \\
&=\mathbb{E}\left[R_{t+1}+\gamma G_{t+1} | S_{t}=s\right] \\
&=\mathbb{E}\left[R_{t+1}+\gamma v\left(S_{t+1}\right) | S_{t}=s\right]
\end{aligned}
$$
价值函数可以分解为两部分：即时收益 Rt+1 和后继状态的折扣价值 γv(St+1)。上式我们称之为**贝尔曼方程（Bellman Equation）**，其衡量了状态价值和后继状态价值之间的关系。

### **马尔可夫决策过程 Markov Decision Process**

相较于马尔可夫奖励过程，马尔可夫决定过程多了一个行为集合A，它是这样的一个元组: <S, A, P, R, γ>。

 S为有限的状态集合，A 为有限的动作集合，P 为状态转移概率矩阵，
$$
\mathcal{P}_{s s^{\prime}}^{a}=\mathbb{P}\left[S_{t+1}=s^{\prime} | S_{t}=s, A_{t}=a\right]
$$

$$
\mathcal{R}_{s}^{a}=\mathbb{E}\left[R_{t+1} | S_{t}=s, A_{t}=a\right]
$$



**策略（Policy）**定义为给定状态下动作的概率分布：
$$
\pi \left(a | s\right) = \mathbb{P} \left[A_t = a | S_t = s\right]
$$
