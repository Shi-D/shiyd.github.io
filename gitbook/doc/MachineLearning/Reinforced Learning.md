# Reinforced Learning

## 强化学习分类

### 分类1

* 不理解环境 Model-Free RL
  * Q-learning
  * Sarsa
  * Policy Gradients

* 理解环境 Model-Based RL
  * 先对真实世界建模
  * 再使用 Model-Free RL 的方法进行玩耍

### 分类2

* 基于概率
  * Policy Gradients
* 基于价值
  * Q-learning
  * Sarsa
* 结合概率与价值
  * Actor-Critic

### 分类3

* 回合更新 Monte-Carlo update
  * 一整个回合结束后，再进行更新
* 单步更新 Temporal-Difference update
  * 游戏开始到结束对过程中一直在更新
  * 更有效，目前主流

### 分类4

* 在线学习
  * Sarsa

* 离线学习
  * Q-learning
  * Deep Q network

