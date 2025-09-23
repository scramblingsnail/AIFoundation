# 注意力层

## **基本计算公式**

将Features $X$（来源可能不同）经过线性变换，得到 $Q$，$K$，$V$。

$$
Q = XW_Q \\
K = XW_K \\
V = XW_V 
$$

对 $Q$，$K$，$V$ 进行计算。具体来说，是对$Q$，$K$求各Token之间的相关性，再用相关性系数对 $V$ 进行加权计算。

$$
Softmax(\frac {Q \cdot K^T} {\sqrt{d}}, dim=-1) \cdot V
$$

这里的scaling $\sqrt{d}$ 解释一下，假设神经网络的特征数据分布服从正态分布（这是在神经网络权重初始化、梯度分析等方面常见的假设）。
即 $E(x) = 0$， $\sigma (x) = 1$。而注意力矩阵 $A=QK^T$ 中每一个元素是服从正态分布的两个 $d$ 维向量的点积：

$$
A_{ij} = \sum_l Q_{il} * K_{lj}
$$

根据期望、方差的公式：

$$
E(Q_{il} * K_{lj}) = E(Q_{il}) * E(K_{lj}) \\
Var(Q_{il} * K_{lj}) = Var(Q_{il}) * Var(K_{lj}) + Var(Q_{il}) \times E(K_{lj})^2 + Var(K_{lj}) \times E(Q_{il})^2
$$

可得 $E(A_{ij}) = 0$，$Var(A_{ij}) = d$，为了保证 $A$ 也满足正态分布，故乘以scaling因子 $\sqrt{d}$ 。

## **Multi-head Attention**
多头注意力有 $H$ 个注意力头，每个包含一组 $Q$，$K$，$V$，当然它们的维度是 $\frac {d} {H}$。
多个注意力头的结果拼接后形成完整输出。

## **Grouped Query Attention**
顾名思义，为了减少 $K$，$V$ 的计算量与存储，将多个Query分为一组，对应相同的 $K$，$V$。

## **Multi-Head latent Attention**
DeepSeek-v2提出的MLA，同样是为了减少 $K$，$V$ 的计算量与存储，将 $W_K$，$W_V$ 均替换为两个低秩矩阵
（一个将Hidden State变换为Latent向量，一个将Latent向量上变换为$K$或$V$）。不再存储 $K$，$V$，而是存储相应的Latent向量，
在需要取用时，基于Latent向量进行线性变换即可。

## **Linearized Attention**
这是一种近似方法，用两个核函数 $h(Q) \cdot h(K^T)$ 去近似 $Softmax(\frac {Q \cdot K^T} {\sqrt{d}})$，这样注意力计算就被简化成：

$$
h(Q) \cdot h(k^T) \cdot V
$$

这样我们就可以将 $Q$ 与 $K$ 解耦，转而先计算 $h(K^T) \cdot V$ 并存储。
在有新Token生成时，我们不再需要重新计算一遍 $Softmax(QK^T)$，而是只需要对存储的 $h(K^T) \cdot V$进行简单的更新，随后再计算与 $Q$ 的点积即可。

## **Attention Mask**
Attention矩阵 $A$ 代表序列中各个Token之间的相关性，通过引入一些先验知识，我们可以通过控制 $A$ 中各元素的取值，来影响各Token之间的相关性。

可以这样理解，$A$ 代表各Token之间在时序维度上的有向全连接图，Attention Mask则是对这张图进行剪枝。

最常用的Attention Mask就是Decoder里的Causal Mask，该Mask保证只有从时序靠前的Token指向时序靠后的Token的连接。
由于 $A$ 的第一个维度是Query相关的输出维度（代表有向连接的终点），第二个维度是Value相关的输入维度（代表有向连接的起点），
因此 Causal Mask $A$ 是一个下三角矩阵。

基于Causal Mask的特点，在Decoding阶段时，只需要计算当前Token的Query与存储的 KV Cache之间的注意力即可（对每一层Decoder Layer均是如此）。

还有一些其它Pattern的Attention Mask，它们均对应于一定的先验假设，例如只有对角线附近元素有效的 $A$ 对应于Token之间的局域相关性假设。
这种通过限制Attention Mask来减少计算量的方式，可能会引入Bias。