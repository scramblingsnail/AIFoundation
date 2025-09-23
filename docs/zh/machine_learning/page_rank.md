# **Page Rank**


## **待解决问题**
给定一张有向图（如互联网），预测每个节点的访问热度，并排序。

## **解决方案**
设各节点以均匀的概率 $\frac {1} {n}$ 访问它所能到达的其它节点（其出度为 n），则在这张图上随机游走的过程是一个马尔可夫过程。可以写出其转移矩阵 $H$ 。

达到平稳分布时：

$$
H \cdot \overrightarrow p = \overrightarrow p
$$

但是有平稳分布的条件是，该马尔科夫链非周期，且强连通。有时该有向图不符合该条件。

给该有向图加上强连通条件：

$$
\overrightarrow {p_1} = (1 - \alpha) H \cdot \overrightarrow {p_0} + \alpha \times \frac {1} {N} \overrightarrow 1
$$

### **迭代法**
根据上式迭代计算 概率分布，直至概率分布趋于平稳。

### **近似-求主特征向量法**
近似：

$$
\overrightarrow {p_1} = \{ (1 - \alpha) H + \alpha \frac {1} {N} E \} \cdot \overrightarrow {p_0}
$$

其中 $E$ 为全1矩阵，求上式矩阵的主特征值（绝对值最大）与主特征向量即可。