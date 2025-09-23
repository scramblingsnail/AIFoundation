# **Layer Normalization**


## **定义**
不同于Batch Normalization， Layer Normalization 是对Embedding的特征维度进行标准化，且不记录每批次训练样本的均值与标准差。

例如，输入数据尺寸为 $(N, L, H, d_{model})$ ，LN对最后一个维度进行标准化，再对该维度进行仿射。

## **Pre-Norm & Post-Norm**
Pre-Norm 与 Post-Norm 的区别在于残差连接中，恒等映射项的来源位置。如果恒等映射项来源于LN层之前，即Pre-Norm；如果来源于LN之后，即Post-Norm。

经典Transformer结构中采用的是Post-Norm，现在多数模型采用Pre-Norm。

关于对Pre-Norm与Post-Norm的分析见：

[👉👉👉 On Layer Normalization in the Transformer Architecture 👈👈👈](https://dl.acm.org/doi/pdf/10.5555/3524938.3525913)

## **变种**

### **RMS Norm**
为了减少计算量，省去均值、标准差的计算过程，直接除去输入数据的均方根，再进行仿射。

Qwen3, Llama4用的LN都是这个。

### **L2 Norm**
直接除去均方根，不进行仿射。Llama4 对 $Q$，$K$ 进行标准化时用的就是这个。