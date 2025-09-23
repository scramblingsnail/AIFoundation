# **Mixture of Experts**

## **基本概念**
将FNN替换为多个隐藏层尺寸较小的FNN（Experts），再引入一个 Router （单层线性变换）生成各个Token ID对于各个Expert的相关性分数，
对每个Token ID，选出 Top-k 个Experts。将对应Token ID的Features送入相应的Expert FNN进行计算。最后将不同Experts对相同Token ID的处理
结果进行聚合（相加）。

优点：减少FNN计算量。