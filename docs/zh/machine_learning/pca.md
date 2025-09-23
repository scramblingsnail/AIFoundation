# **主成分分析**


## **待解决问题**
给定N个样本，H个特征维度，将这些样本数据降维至K个维度。这些维度的特点是：

1. 将样本投影至第一个维度，数据之间的方差最大。投影至该维度后的数据称为第一主成分。
2. 第二个维度与第一个维度正交，且将样本数据投影至这个维度，数据方差最大（除了它之前的维度）。第二主成分。
3. 依次类推 ...

## **解决方案**
求解主成分，即求解样本数据不同维度之间的规范协方差矩阵的特征值与特征向量。具体证明参见《统计学习方法》，李航。

设样本数据为 $D$ ，尺寸为 $N \times K$ 。将样本数据按特征维度进中心化：

$$
X = D - mean(D, axis=0)
$$

不同维度间的协方差矩阵为：

$$
Cov = \frac {1} {N-1} X^T \cdot X
$$

求解协方差矩阵的特征值与特征向量，并按特征值大小排序。特征向量就是各正交投影维度，特征值就是在相应投影维度上的样本方差。

## **代码**
<details>
<summary>主成分分析</summary>
```python
import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, K: int):
        self.data = self.load_data()
        self.normalize()
        self.K = K

    def load_data(self):
        N, H = 100, 10
        std_list = [100, 2, 1]
        data = np.random.randn(N * H).reshape(N, H)
        for idx in range(len(std_list)):
            data[:, idx] *= std_list[idx]
        data[:, len(std_list):] *= 0.1
        return data

    def normalize(self):
        mean = np.mean(self.data, axis=0, keepdims=True)
        self.data = self.data - mean

    def pca(self):
        cov = self.data.T @ self.data
        # linear algebra -> eig
        eig_vals, eig_vectors = np.linalg.eig(cov)
        # 按特征值大小排序
        sorted_idx_vals = sorted(enumerate(eig_vals), key=lambda x: x[1], reverse=True)
        indices = [x[0] for x in sorted_idx_vals]
        vals = [x[1] for x in sorted_idx_vals]
        # 列向量是特征向量
        eig_vectors = eig_vectors[:, indices]
        # H * K
        transform = eig_vectors[:, :self.K]
        # N * K
        trsf_data = self.data @ transform

        plt.scatter(trsf_data[:, 0], trsf_data[:, 1])
        plt.show()


if __name__ == "__main__":
    worker = PCA(K=2)
    worker.pca()
```

</details>