# **FNN**

## **结构**

**基本**

两层前向神经网络，隐藏层大小通常为 $4d_{model}$

**Gated**

两层前向神经网络，带一个Gate层（激活函数为类Sigmoid函数，如SiLU），Gate层输出与隐藏层输出进行Hardmard积之后，再进入输出层。
