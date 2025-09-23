# **Decoding**


## **待解决问题**
在进行自回归生成时，如何选择最优生成序列

## **确定性生成 -- Greedy Search**
在每一步选择当前概率最大的Token。

## **确定性生成 -- Beam Search**
维护一个容量为 N 的Buffer，记录N条条件概率最高的序列。

每次生成时，根据这N条序列生成新Token，并在这些新序列中选出概率最高的N条序列，并更新 Buffer。

## **不确定性生成 -- Multinomial Sampling**
如何从候选Token中选出下一个生成的Token。

**Top-k Sampling**

只从概率最高的前 K 个Token中进行采样（概率相应地归一化）。

**Top-p Sampling**

将各Token的概率从高到低排序，计算累计概率。只在累计概率小于 $p$ 的Token中进行采样。

**Temperature**
即 Softmaxt 函数中的温度项 T:

$$
Softmax(x; T) = \frac {e^{\frac {x_i} {T}}} {\sum_i e^{\frac {x_i} {T}}}
$$

$T$ 越大，logit x 的不同取值间的差异就越小；生成更具随机性。

$T$ 越小，logit x 的不同取值间的差异就越大；生成过程更加确定。