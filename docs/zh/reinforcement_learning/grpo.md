# **GRPO**

## **论文地址**

[👉👉👉 Group Relative Policy Optimiaztion 👈👈👈](https://arxiv.org/pdf/2402.03300)

## **原理**
GRPO是为LLM的RL fine-tuning阶段优化设计的PPO变种算法，其核心在于使用蒙特卡洛估计法替代PPO算法中的时序差分估计法与Critic网络，
以减少训练成本（由于Critic网络参数量与待微调模型相当，此时使用并训练Critic网络的成本远高于蒙特卡洛采样法多次采样的成本）。

此外还添加了KL散度的Penalty，限制微调模型相对于预训练模型的输出分布差异。

## **伪代码**