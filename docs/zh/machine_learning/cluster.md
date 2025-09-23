# **聚类**

## **待解决问题**
给定N个样本，特征维度数为H，根据特定距离度量将这些样本聚类。

## **距离度量**

**闵可夫斯基距离：**

$$
L^p(\overrightarrow {X_1}, \overrightarrow {X_2}) = (\sum_{i=1}^H |\overrightarrow {X_1}(i) - \overrightarrow {X_2}(i)|^p)^{\frac {1} {p}}
$$

p = 1 → 曼哈顿距离；

p = 2 → 欧式距离；

p = $\infty$ → 切比雪夫距离

**余弦相似度：**

$$
R(\overrightarrow {X_1}, \overrightarrow {X_2}) = \frac {\overrightarrow {X_1} \cdot \overrightarrow {X_2}} {|\overrightarrow {X_1}| |\overrightarrow {X_2}|}
$$

## **类间距离度量**

1. 两类所有样本Pair的最短距离
2. 两类的中心点的最长距离
3. 两类所有样本Pair的最长距离

## **层次聚类**
聚合式聚类，最初将N个样本分为N类，每次选择距离最近的两个类进行合并（分词算法中的BPE也可视作一种聚类）。
直到达到停止条件（如类的数量减少到一定值，或者最短类间距离增加到一定值）。

### **代码**
<details>
<summary>层次聚类</summary>
```python
import numpy as np
import matplotlib.pyplot as plt


class Cluster:
    def __init__(self):
        self.data = self.load_data()
        self.dist_matrix = self.compute_dist_matrix()

    def load_data(self):
        # N, K
        N, K = 50, 2
        data_1 = np.random.randn(N * K).reshape((N, K)) - 10
        data_2 = np.random.randn(N * K).reshape((N, K)) + 10
        data_3 = np.random.randn(N * K).reshape((N, K)) - 20
        data_4 = np.random.randn(N * K).reshape((N, K)) + 20
        data = np.concatenate((data_1, data_2, data_3, data_4), axis=0)
        return data

    def _dist(self, i, j):
        dist = np.power(self.data[i] - self.data[j], 2)
        dist = np.sqrt(np.sum(dist))
        return dist

    def compute_dist_matrix(self):
        num = self.data.shape[0]
        dist_matrix = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dist_matrix[i, j] = self._dist(i, j)
                dist_matrix[j, i] = dist_matrix[i, j]
        return dist_matrix

    def _dist_between_cluster(self, c1: list, c2: list):
        min_dist = np.inf
        for i in c1:
            for j in c2:
                min_dist = min(min_dist, self.dist_matrix[i][j])
        return min_dist

    def _each_cluster(self, cluster_list: list):
        c_num = len(cluster_list)
        min_dist = np.inf
        min_pair = None
        for i in range(c_num):
            for j in range(i + 1, c_num):
                dist = self._dist_between_cluster(cluster_list[i], cluster_list[j])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)
        pop_c = cluster_list.pop(min_pair[1])
        cluster_list[min_pair[0]].extend(pop_c)
        return cluster_list

    def cluster(self):
        num = self.data.shape[0]
        cluster_list = [[d_idx] for d_idx in range(num)]

        c_num = len(cluster_list)
        while c_num > 4:
            cluster_list = self._each_cluster(cluster_list)
            c_num = len(cluster_list)
        return cluster_list


if __name__ == "__main__":
    worker = Cluster()
    res = worker.cluster()
    c_list = ["yellow", "blue", "red", "black"]
    for c_idx, idx_list in enumerate(res):
        for d_idx in idx_list:
            point = worker.data[d_idx]
            plt.scatter(*point, c=c_list[c_idx])
    plt.show()
    print(res)
```

</details>

## **K均值聚类**
K均值聚类是一种基于划分的启发式方法，不保证收敛至全局最优。它将样本集合聚为 K 个类，每个类均有一个类中心（该类样本的均值）。

从 K 个初始类中心开始，每次遍历所有样本，根据样本到这K个类中心距离重新划分为K个类，并更新K个类中心。
重复划分→确定类中心的过程，直到类的划分方式不再变化为止。

可见K均值聚类的效果依赖于初始类中心位置的选取，可以预先用其它聚类方法（如层次聚类）获得一个比较好的初始划分。

### **代码**

<details>
<summary>K均值聚类</summary>

```python
import numpy as np
import matplotlib.pyplot as plt


class Cluster:
    def __init__(self):
        self.data = self.load_data()
        self.dist_matrix = self.compute_dist_matrix()

    def load_data(self):
        # N, K
        N, K = 50, 2
        data_1 = np.random.randn(N * K).reshape((N, K)) - 10
        data_2 = np.random.randn(N * K).reshape((N, K)) + 10
        data_3 = np.random.randn(N * K).reshape((N, K)) - 20
        data_4 = np.random.randn(N * K).reshape((N, K)) + 20
        data = np.concatenate((data_1, data_2, data_3, data_4), axis=0)
        return data

    def _dist(self, i, j):
        dist = np.power(self.data[i] - self.data[j], 2)
        dist = np.sqrt(np.sum(dist))
        return dist

    def compute_dist_matrix(self):
        num = self.data.shape[0]
        dist_matrix = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dist_matrix[i, j] = self._dist(i, j)
                dist_matrix[j, i] = dist_matrix[i, j]
        return dist_matrix

    def _dist_between_cluster(self, c1: list, c2: list):
        min_dist = np.inf
        for i in c1:
            for j in c2:
                min_dist = min(min_dist, self.dist_matrix[i][j])
        return min_dist

    def _each_cluster(self, cluster_list: list):
        c_num = len(cluster_list)
        min_dist = np.inf
        min_pair = None
        for i in range(c_num):
            for j in range(i + 1, c_num):
                dist = self._dist_between_cluster(cluster_list[i], cluster_list[j])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)
        pop_c = cluster_list.pop(min_pair[1])
        cluster_list[min_pair[0]].extend(pop_c)
        return cluster_list

    def cluster(self):
        num = self.data.shape[0]
        cluster_list = [[d_idx] for d_idx in range(num)]

        c_num = len(cluster_list)
        while c_num > 4:
            cluster_list = self._each_cluster(cluster_list)
            c_num = len(cluster_list)
        return cluster_list


class KMeanCluster:
    def __init__(self, K: int):
        self.K = K
        self.data = self.load_data()

    def load_data(self):
        # N, H
        N, H = 50, 2
        data_1 = np.random.randn(N * H).reshape((N, H)) - 10
        data_2 = np.random.randn(N * H).reshape((N, H)) + 10
        data_3 = np.random.randn(N * H).reshape((N, H)) - 20
        data_4 = np.random.randn(N * H).reshape((N, H)) + 20
        data = np.concatenate((data_1, data_2, data_3, data_4), axis=0)
        return data

    def _is_same_cluster(self, c_list_1, c_list_2):
        assert len(c_list_1) == len(c_list_2)
        for idx, c1 in enumerate(c_list_1):
            if set(c1) != set(c_list_2[idx]):
                return False
        return True

    def _dist(self, point_1, point_2):
        return np.sqrt(np.sum(np.power(point_1 - point_2, 2)))

    def _compute_mean(self, idx_list):
        num = len(idx_list)
        if num == 0:
            raise ValueError("Empty cluster")
        mean = np.zeros(self.data.shape[-1])
        for d_idx in idx_list:
            mean += self.data[d_idx]
        mean /= num
        return mean

    def _each_cluster(self, mean_list):
        cluster_list = [[] for _ in range(self.K)]
        for d_idx in range(self.data.shape[0]):
            min_dist = np.inf
            belongto = None
            for k_idx in range(self.K):
                k_dist = self._dist(mean_list[k_idx], self.data[d_idx])
                if k_dist < min_dist:
                    min_dist = k_dist
                    belongto = k_idx
            cluster_list[belongto].append(d_idx)
        # 计算新均值
        for k_idx in range(self.K):
            mean_list[k_idx] = self._compute_mean(cluster_list[k_idx])
        return cluster_list, mean_list

    def cluster(self):
        # K * H
        mean_list = self.data[::50, :]
        cluster_list = [[] for _ in range(self.K)]
        max_iter = 100
        count = 0
        while count < max_iter:
            count += 1
            new_cluster_list, mean_list = self._each_cluster(mean_list)
            if self._is_same_cluster(new_cluster_list, cluster_list):
                cluster_list = new_cluster_list
                break
        print(count)
        cluster_list = new_cluster_list
        return cluster_list


if __name__ == "__main__":
    worker = KMeanCluster(4)
    res = worker.cluster()
    c_list = ["yellow", "blue", "red", "black"]
    for c_idx, idx_list in enumerate(res):
        for d_idx in idx_list:
            point = worker.data[d_idx]
            plt.scatter(*point, c=c_list[c_idx])
    plt.show()
```

</details>