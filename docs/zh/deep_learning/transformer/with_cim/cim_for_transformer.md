# **基于CIM加速Transformer推理的讨论**

## **固定权重部分**
LLM中的所有包含固定权重的模块都可以方便地使用CIM进行推理加速。（只要存储容量足够大，就可以不进行重新编程）。例如以下模块

### **生成QKV的线性变换**

$$
Q,K,V = X \cdot W_Q,W_K,W_V
$$

将 $W_Q,W_K,W_V$ 存入CIM阵列中，输入 $X$ 即可。若 $X$ 尺寸为 $(B,T,d)$，需要进行 $BT$ 次操作，因此可以将这些矩阵复制多份，
用多个CIM阵列并行处理。

### **MoE模块中的Router**

$$
X \cdot W_R
$$

$W_R$ 权重尺寸为 $d \times N_{experts}$, 同样可以复制多份，并行处理。

### **MoE模块中的FNN**
以Qwen3 MoE为例，每个Expert的FNN包含三个权重矩阵：$W_{up}$，$W_{Gate}$，$W_{Down}$;
尺寸分别为 $d_{model} \times d_{hidden}$，$d_{model} \times d_{hidden}$，$d_{hidden} \times d_{model}$。

### **自注意力模块**

**Prefill阶段**

设Prefill长度为 $L_{pre}$，Batch size为 $B$。
将线性变换产生的 $K$，$V$ 存储在 $2B$ 个 CIM阵列中，这里需要进行 $O(B \times L_{pre} \times d_{model})$ 次编程操作。

接下来计算 $Softmax(\frac {Q \cdot K^T} {\sqrt {d}}) \cdot V$，需要进行$O(BT)$次点乘操作，配合数字计算电路完成Softmax的计算。

**Decoding阶段**

在Decoding阶段，每次生成只需要计算当前Token的Query即可，也就是进行$O(B)$次点乘操作。
需要注意的是，随着Decoding的进行，$K$，$V$矩阵的尺寸在不断增长，因此每次都需要进行 $O(B \times d_{model})$ 次编程操作。这对于CIM阵列的编程速度
提出了一定要求。


