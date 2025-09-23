# **隐马尔可夫模型**

## **待解决问题**
观测到一个状态序列，用一个隐藏的马尔可夫链建模状态间的转移关系。

该马尔可夫链有 M 个内部状态，内部状态转移矩阵为 $H$ (尺寸 $M \times M$ )。共有 N 个可观测状态，从各内部状态观测到各观测状态的概率矩阵为 $B$ (尺寸 $N \times M$ )。

已知观测序列，与马尔科夫链初始内部状态 $\overrightarrow {P_0}$，求最有可能的内部状态序列。

## **解决方案**

### **已知观测序列情况下的某一步内部状态的前向概率**

设观测状态的索引序列为 $o^{(1)}, o^{(2)}, ...$

$$
P^{(t+1)} = H \cdot p^{(t)} \odot B[o^{(t)}]^T
$$

### **近似算法**
贪心策略

计算已知观测序列情况下的前向概率（各内部状态的概率），每次选择概率最大的那个内部状态加入预测的内部序列。

### **维特比算法**
动态规划 + 束搜索

总共M个内部状态，最后一个内部状态有 M 种可能，对每一个最后状态，其前面的序列都有一个最优选择，因此，到最后一个状态时，最多有M个选择。
所以只要进行宽度为M的束搜索就行，buffer里面维护的是：到某一步时，达到M个内部状态各自的最优序列。

例如，在初始时，K个最优序列就是K个内部状态；在第二步，对每个状态计算到达该状态的最优前向概率（从第一步的状态里选），并更新K-buffer。
直到最后一个状态，从这 K 个状态（以及到达它们的最优序列）中选择出，能够产生最后观测状态，且概率最大的那条序列，它就是最优的内部状态序列。

## **代码**

<details>
<summary>预测隐马尔可夫链内部状态</summary>
```python
import numpy as np


class Markov:
    def __init__(self):
        self.H, self.B = self.load_transfer_matrices()

    def load_transfer_matrices(self):
        # M个hidden state，N个观测状态
        # M * M； H 的行对应输出，列对应输入，同一列的所有概率加起来等于 1
        M, N = 10, 5
        H = np.random.randn(M * M).reshape((M, M))
        H = np.abs(H)
        row_sum = np.sum(H, axis=0)
        H = H / row_sum

        # N * M：B 的行为输出，列为输入，同一列的所有概率加起来等于1
        B = np.random.randn(N * M).reshape((N, M))
        B = np.abs(B)
        b_row_sum = np.sum(B, axis=0)
        B = B / b_row_sum
        return H, B

    def compute_forward_prob(self, init_p, step_num: int, ob_indices):
        # p: (M * 1)
        p = init_p
        for i in range(step_num):
            ob_i = ob_indices[i]
            p = self.B[ob_i:ob_i+1, :].T * self.H @ p
        return p

    def _step_forwad_p(self, before_p, ob_idx):
        new_p = self.B[ob_idx: ob_idx+1, :].T * self.H @ before_p
        return new_p

    def optimal_hidden(self, init_p, ob_indices: list):
        # 近似算法。
        L = len(ob_indices)
        # M * 1
        p = init_p
        hidden_list = []
        for step_idx in range(L):
            ob_idx = ob_indices[step_idx]
            ob_p = p * self.B[ob_idx: ob_idx+1, :].T
            max_p = max(list(ob_p.reshape(-1)))
            max_idx = list(ob_p.reshape(-1)).index(max_p)
            hidden_list.append(max_idx)
            p = self.H @ ob_p
        return hidden_list

    def optimal_hidden_witb(self, init_p, ob_indices: list):
        # beam buffer
        L = len(ob_indices)
        M = self.H.shape[0]

        k_beams = [[i, ] for i in range(M)]
        # M * 1
        k_max_p = init_p

        for step_idx in range(1, L):
            new_k_beams = [[] for _ in range(M)]
            new_k_max_p = np.zeros(k_max_p.shape)
            for hid_idx in range(M):
                before_ob = ob_indices[step_idx - 1]
                # 前向概率要计入观测到已观测值的概率。
                ob_prob = self.B[before_ob: before_ob + 1, :].T
                to_k_probs = k_max_p * ob_prob * self.H[hid_idx: hid_idx+1, :].T
                max_to_k_p = max(list(to_k_probs.reshape(-1)))
                # 更新最优概率
                new_k_max_p[hid_idx] = max_to_k_p
                max_before_hid = list(to_k_probs.reshape(-1)).index(max_to_k_p)
                # 记录到当前 hid_idx 的最优路径
                to_k_list = k_beams[max_before_hid].copy()
                to_k_list.append(hid_idx)
                new_k_beams[hid_idx] = to_k_list
            k_beams = new_k_beams
            k_max_p = new_k_max_p
        # 处理最后的K个概率
        ob_prob = self.B[-1:, :].T
        ob_final_p = k_max_p * ob_prob
        ob_max_p = np.max(ob_final_p)
        max_hid_idx = list(ob_final_p).index(ob_max_p)
        max_hid_list = k_beams[max_hid_idx]
        return max_hid_list


if __name__ == "__main__":
    worker = Markov()

    M, N = 10, 5
    init_prob = np.random.randn(M).reshape((M, 1))
    init_prob = np.abs(init_prob)
    init_prob = init_prob / np.sum(init_prob)

    ob_indices = list(np.random.randint(0, N, (6, )))

    h_list = worker.optimal_hidden(init_p = init_prob, ob_indices=ob_indices)
    print(h_list)
    h_list = worker.optimal_hidden_witb(init_p = init_prob, ob_indices=ob_indices)
    print(h_list)
```
</details>