# **逻辑回归**

## **待解决问题**
二分类问题。给N个训练样本，特征维度数为K。

## **解决方案**
使用单层线性变换（(K, 1)）+ sigmoid 函数作为模型，输出分类为 1 的概率（Label是 1 或 0）。

设输入的 $1 \times K$ 特征数据为 $\overrightarrow X$ ，线性变换矩阵为 $W$ 。输出为：

$$
p = sigmoid(\overrightarrow X \cdot W)
$$

损失函数为交叉熵：

$$
\min_W L(W) = \sum_i H(y_i, p_i) = - \sum_i y_i \log p_i + (1 - y_i) \log (1 - p_i)
$$

梯度的解析式：

$$
\frac {\partial L} {\partial W} = \sum_i \frac {\partial L} {\partial p_i} \frac {\partial p_i} {\partial W}
$$

其中

$$
\frac {\partial L} {\partial p_i} = \frac {-1} {In2} (\frac {y_i} {p_i} + \frac {1 - y_i} {p_i - 1}) = \frac {1} {In2} \frac {{y_i - p_i}} {{p_i (p_i - 1)}}
$$

$$
\frac {\partial p_i} {\partial W} = (p_i - p_i^2) \overrightarrow {X}
$$

于是

$$
\frac {\partial L} {\partial W} = \sum_i \frac {1} {In2} (p_i - y_i) \overrightarrow X
$$

使用梯度下降法，沿该梯度反方向以一定学习步长更新 $W$ 即可。

## **代码**
<details>
<summary>梯度下降法求逻辑回归</summary>
```python
import numpy as np


class LogisticReg:
    def __init__(self, lr: float):
        self.data = self.load_data()
        self.W = np.random.randn(self.data.shape[-1] - 1).reshape(-1, 1)
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def load_data(self):
        N, K = 1000, 10
        pos_features = np.random.randn(N * K).reshape((N, K)) + 1
        pos_labels = np.ones((N, 1))

        neg_features = np.random.randn(N * K).reshape((N, K)) - 1
        neg_labels = np.zeros((N, 1))

        features = np.concatenate((pos_features, neg_features), axis=0)
        labels = np.concatenate((pos_labels, neg_labels), axis=0)
        data = np.concatenate((features, labels), axis=-1)
        return data

    def forward_backward_update(self):
        x = self.data[:, :-1]
        # x: N * K; W: K * 1
        p = self.sigmoid(x @ self.W)
        # N * 1
        y = self.data[:, -1:]
        # N * k
        grad = (p - y) * x
        grad = np.mean(grad, axis=0, keepdims=True)
        grad = grad / np.max(np.abs(grad))
        # update
        self.W = self.W - self.lr * grad.T
        # print(self.W)

    def eval(self):
        x = self.data[:, :-1]
        p = self.sigmoid(x @ self.W)
        y = self.data[:, -1:]
        predict = (p > 0.5) * 1
        correct = np.sum(np.equal(predict, y))
        acc = correct / y.shape[0]
        print("ACC: ", acc)

    def train(self):
        for epoch in range(100):
            self.forward_backward_update()


if __name__ == "__main__":
    log = LogisticReg(lr=0.1)
    log.train()
    log.eval()
```

</details>