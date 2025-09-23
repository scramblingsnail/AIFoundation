# 强化学习方法的分类

基本上所有的强化学习方法都可以视作为一种广义的策略迭代过程，包含两部分：

- 策略评估：即更新价值函数，用于评价策略的好坏程度。
- 策略提升：即根据价值函数去改变策略，使策略更优。


## **基于价值函数的方法（Value-based）**
在基于价值函数的方法中，策略是依附于价值函数的。

例如将策略选择为：$a^*=argmax_{a} Q(s, a)$ (Q Learning)，
或者策略的概率分布依赖于价值函数（如SAC假定动作概率分布服从以$-Q(s, a)$为能量项的玻尔兹曼分布）。因此，在完成策略评估过程后，价值函数的更新
同时完成了策略提升的过程。

回顾贝尔曼最优方程：

$$
Q^*(s, a) ← r + Q^*(\hat s, a_{max}) 
$$

Value-based 方法正是基于上式进行价值函数更新，可见在Value-based方法中，“价值函数”指的是最优价值函数。
而最优价值函数是不依赖于特定策略的，因此Value-based方法通常是异策略（off-policy）的，可以使用任意的Transition数据 $(s, a, r, \hat s)$ 去迭代估计 $Q^*(s, a)$。
通常可以使用Relay buffer来储存已采样的所有数据，降低采样成本。

## **基于策略的方法（Policy-based）**
在Policy-based的方法中，策略是独立存在的模型，并且直接求取回报期望$R$对于策略模型参数$\theta$的梯度，以更新策略。

策略梯度：

$$
\nabla_{\theta} R(\theta) = \nabla_{\theta} E_{\pi(\theta)} \sum_{t=1}^T r_t \gamma^{t-1}
$$

可以证明在$\gamma$接近1的情况下，策略梯度为：

$$
\nabla_{\theta} R(\theta) = E_{\pi(\theta)} Q_{\pi(\theta)}(s, a) \nabla_{\theta}\log \pi(a|s;\theta)
$$

可见Policy-based方法同样需要估计价值函数（这里的价值函数依赖于策略，非最优价值函数）。

为了避免Overestimation，一般在估计所得Q(s,a)的基础上减去一个动作无关的值，通常减去状态价值函数$V(s)$。

$$
\nabla_{\theta} R(\theta) = E_{\pi(\theta)} A_{\pi(theta)}(s, a) \nabla \log \pi(a|s;\theta) \\
A(s, a) = Q(s, a) - V(s)
$$

$A(s, a)$为优势函数，指在采取动作 $a$ 后相比于当前状态增加的累计回报期望。

## **Actor-Critic方法**
结合Value-based与Policy-based方法，并显式地分别估计Value与Policy。
其中估计Value的模型称为Critic，产生Policy的模型称为Actor。

Actor-Critic方法显式地进行策略评估与策略提升过程，分别优化Critic与Actor。