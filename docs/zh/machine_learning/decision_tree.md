# **决策树**

## **待解决问题**
现有N个训练样本，每个样本K个特征，可能类别有C种。
给定测试样本，判断测试样本的类别。

## **解决方案 -- 构建决策树**
决策树依靠对样本每个特征取值的判断来对样本进行分类：如判断 特征1 → 判断特征2 → ... → 判断特征K → 剩下的训练样本，根据训练样本的类别进行多数表决。

可见特征判断的次序对于决策树的分类结果十分重要，如何选取特征的次序呢？或者说，如何比较不同特征维度对于分类的重要程度呢？一个好的特征应当大幅度减少
样本数据对于标签分类的不确定性，这个不确定性我们可以用信息熵来衡量：

给定一部分样本数据 $D$，这些数据中各个标签类别的样本数量占总样本数量的比例分别为 $p_1, p_2, ..., p_K$ ，则这个样本集合对于标签的信息熵（经验熵）为：

$$
H(D) = \sum_{i=1}^K -p_i \log p_i
$$

一个好的特征应该：按这个特征的取值将样本数据划分为一系列子集，这些子集的经验熵的加权和（条件熵）应该尽可能小（对于标签的不确定性尽可能小）。

假设现在按照特征维度 $F_l$ 进行划分，$F_l$ 的 $M_l$ 个取值将样本数据划分为 $M_l$ 个子集 $D_i$ ，每个子集有 $|D_i|$ 个数据，则划分之后的条件熵为：

$$
H(D|F_l) = \sum_{i=1}^{M_l} \frac {|D_i|} {|D|} H(D_i)
$$

定义按某特征 $F_l$ 划分带来的 **信息增益** 为：

$$
g(D, F_l) = H(D) - H(D|F_l)
$$

即按某特征 $F_l$ 划分后熵的减少量，信息增益非负。由此，我们就可以根据信息增益的大小判断各特征的重要程度，并将信息增益大的特征放在次序靠前的位置。

这样，有了特征次序选择准则之后，我们可以根据样本数据集合构建一棵决策树，该决策树的特点是：每一个结点对应一个特征的划分，该结点的子结点对应划分后的样本数据子集合。

可见每个结点应该存储的信息包括：1. 该结点的样本数据集合； 2. 该结点对应的特征维度； 3. 该结点的各个子结点，以及对应的特征取值。

可以递归地构建决策树，在每一步的工作包括：

1. 特征维度选取：根据当前的样本集合，遍历每一个可能的特征维度，计算按该特征维度划分后的信息增益。选取信息增益最大的特征作为划分特征。
2. 按选定特征划分样本集合，使用划分后的子集合与移除当前特征后的可选特征集合，递归地构建子树。
3. 结束条件：
> 1. 样本集合中所有数据是同一类，此时信息熵最小，将本叶子结点的类别标记样本类别；
> 2. 没有可用于划分的特征维度了，按多数表决原则标记本叶子结点类别 
> 3. 其它条件，如样本集合信息熵足够小等等。


### **决策树的剪枝**
自底向上地递归遍历决策树（后序遍历），每访问一个非叶子结点，判断如果将该结点设置为叶子结点，决策树的相关指标是否会有提升。
这里需要一个判断标准：如在特定数据上的推理性能指标、（信息熵+子节点个数）等等。

1. 若当前结点是叶子结点，返回。
2. 递归地对当前结点的子结点进行剪枝。
3. 将当前结点设为叶子结点，并计算相关的评价指标，如果相对于剪枝之前有提升，则最终确定将当前结点设为叶子结点，否则还原为非叶子结点。


### **代码**
<details>
<summary>构建决策树以及决策树推理</summary>
```python
import numpy as np
from typing import Optional


class Node:
    def __init__(self, data_indices: tuple, f_idx: Optional[int]=None, is_leaf: bool = False, label: Optional[int] = None):
        self.data_indices = data_indices
        self.f_idx = f_idx
        self.is_leaf = is_leaf
        self.label = label
        self.children = dict()

    def add_children(self, node, f_val):
        # (子节点， 特征取值)
        self.children[f_val] = node


class DecisionTree:
    def __init__(self, name: str = None):
        self.name = name
        # (N, K)
        self.data = self.load_data()

    def load_data(self, ):
        # 获取样本数据 -> (N, K)
        feature_val_min, feature_val_max = 0, 10
        lables_num = 2
        N, K = 1000, 10
        feature_data = np.random.randint(feature_val_min, feature_val_max, (N, K))
        labels = np.random.randint(0, lables_num, (N, 1))
        data_labels = np.concatenate((feature_data, labels), axis=-1)
        return data_labels

    def entropy(self, data_indices: list):
        # 计算样本集合对于标签的信息熵
        p_dict = dict()
        for d_idx in data_indices:
            label = self.data[d_idx, -1]
            label_num = p_dict.get(label, 0)
            p_dict[label] = label_num + 1
        all_num = len(data_indices)
        p_list = np.array([p_dict[key] / all_num for key in p_dict.keys()])
        entropy = np.sum(- p_list * np.log(p_list))
        return entropy

    def child_entropy(self, data_indices: list, f_idx: int):
        # Key: 特征取值； Val: 样本索引列表
        child_idx_list = dict()
        all_num = len(data_indices)
        for d_idx in data_indices:
            f_val = self.data[d_idx, f_idx]
            sub_indices = child_idx_list.get(f_val, [])
            sub_indices.append(d_idx)
            child_idx_list[f_val] = sub_indices
        ent = 0
        child_ent_dict = dict()
        for f_val in child_idx_list.keys():
            each_indices = child_idx_list[f_val]
            each_ent = self.entropy(each_indices)
            child_ent_dict[f_val] = each_ent
            ent += len(each_indices) / all_num * each_ent
        return ent, child_idx_list, child_ent_dict

    def decide_label(self, data_indices: list):
        label_dict = dict()
        max_label = None
        max_count = 0
        for d_idx in data_indices:
            label = self.data[d_idx, -1]
            count = label_dict.get(label, 0) + 1
            label_dict[label] = count
            if count > max_count:
                max_label = label
                max_count = count
        return max_label

    def construct_tree(self, data_indices: list, avail_features: list, data_entropy: float = None):
        # data_indices: 样本的索引集合
        # avail_features: 可用于划分的特征维度索引集合

        if len(data_indices) == 0:
            return None

        if data_entropy is None:
            # 计算样本集合的信息熵
            data_entropy = self.entropy(data_indices)

        # 如果全是同一个类别，熵为0
        if data_entropy == 0:
            node = Node(data_indices=data_indices, is_leaf=True, label=self.data[data_indices[0], -1])
            return node

        # 如果没有足够特征维度
        if len(avail_features) == 0:
            label = self.decide_label(data_indices)
            node = Node(data_indices=data_indices, is_leaf=True, label=label)
            return node

        min_ent = np.inf
        child_indices = None
        child_ent = None
        chose_f = None
        for f_idx in avail_features:
            # 计算按 f_idx 划分的条件熵。
            f_ent, sub_idx_dict, child_ent_dict = self.child_entropy(data_indices=data_indices, f_idx=f_idx)
            if f_ent < min_ent:
                chose_f = f_idx
                min_ent = f_ent
                child_indices = sub_idx_dict
                child_ent = child_ent_dict

        # 构建结点
        node = Node(data_indices=data_indices, f_idx=chose_f)

        # 递归地构建子树
        next_features = avail_features.copy()
        next_features.remove(chose_f)
        for f_val in child_indices.keys():
            child_node = self.construct_tree(data_indices=child_indices[f_val], avail_features=next_features, data_entropy=child_ent[f_val])
            if child_node is not None:
                node.add_children(child_node, f_val)
        return node

    def classify(self, test_data_vec, root: Node):
        if root is None:
            return None

        if root.is_leaf:
            return root.label

        f_idx = root.f_idx
        f_val = test_data_vec[f_idx]
        if f_val not in root.children.keys():
            # 多数表决
            label = self.decide_label(root.data_indices)
            return label
        child = root.children[f_val]
        label = self.classify(test_data_vec, child)
        return label

    def inference(self, test_data, root: Node):
        labels = test_data[:, -1]
        features = test_data[:, :-1]
        N = features.shape[0]
        label_true_n = np.sum(labels == 1)

        outputs = []
        pre_t_l_f, pre_t_l_t, pre_f_l_f, pre_f_l_t = 0, 0, 0, 0

        for d_idx in range(N):
            predict = self.classify(features[d_idx], root)
            label = labels[d_idx]
            outputs.append(predict)
            if predict == 0 and label == 0:
                pre_f_l_f += 1
            elif predict == 0 and label == 1:
                pre_f_l_t += 1
            elif predict == 1 and label == 0:
                pre_t_l_f += 1
            else:
                pre_t_l_t += 1
        assert label_true_n == pre_t_l_t + pre_f_l_t
        # print(pre_t_l_f, pre_t_l_t, pre_f_l_f, pre_f_l_t)
        recall = pre_t_l_t / label_true_n

        if pre_t_l_t + pre_t_l_f == 0:
            precision = 0
        else:
            precision = pre_t_l_t / (pre_t_l_t + pre_t_l_f)
        f1_score = 2 * recall * precision / (recall + precision)

        return f1_score

    def prune(self, node: Node, root: Node):
        # 例如，按照训练数据的 F1 Score 剪枝。
        # 自底向上，递归地判断每个结点的子结点是否有必要剪掉。
        if node.is_leaf:
            return

        # 自底向上递归，后序遍历
        for f_val in node.children.keys():
            child = node.children[f_val]
            self.prune(child, root)

        # 剪枝前推理：
        score_before = self.inference(self.data, root)
        # 剪枝
        node.is_leaf = True
        node.label = self.decide_label(node.data_indices)
        # 剪枝后推理：
        score_after = self.inference(self.data, root)
        if score_after < score_before:
            # 不剪枝，还原
            node.is_leaf = False
            node.label = None
        else:
            print("Pruned ", node.f_idx, node.data_indices)


if __name__ == "__main__":
    d_tree = DecisionTree()
    d_tree.load_data()

    init_indices = list(range(d_tree.data.shape[0]))
    init_features = list(range(d_tree.data.shape[1] - 1))
    t_root = d_tree.construct_tree(init_indices, init_features)
    # 剪枝
    d_tree.prune(t_root, t_root)
    # 推理
    test_data = np.random.randint(0, 10, (200, 10))
    for test_idx in range(test_data.shape[0]):
        res = d_tree.classify(test_data[test_idx], t_root)
        print(f"No. {test_idx} Label: ", res)
```

</details>


## **二叉决策树（分类与回归）的构建与剪枝**
二叉决策树同样是按不同的特征维度依次进行判断，不过这里的判断是一个二值的判断。即对某特征维度找到一个切分点，将该特征维度分为两部分区域，并将属于
这两部分区域的样本数据分别分给两个子结点。这里涉及两个选择：

- 特征维度的选择
- 在该特征维度上的最佳切分点的选择。

这两个选择的评估标准：对于分类决策树，是信息增益、基尼系数等等；对于回归决策树，是两部分数据相对于各自均值的平方损失之和。

**基尼系数：**

$$
Gini(p) = 1 - \sum_i (p_i)^2
$$

其中 $p_i$ 指的是标签为 $i$ 的样本数据占当前样本集合的比例。基尼系数可以视作对信息熵的近似。       

**平方回归损失：**

$$
L = \sum_i (y_i^{(1)} - \bar y^{(1)})^2 + \sum_i (y_i^{(2)} - \bar y^{(2)})^2
$$

其中 $\bar y^{(1)}$ 与 $\bar y^{(2)}$ 分别为两部分样本的输出值均值。在回归决策树推理过程中，输出就是被预测为各叶子结点的输出值均值。

### **代码**

二叉决策树的构建与剪枝与普通决策树大体一致，不过在构建结点的过程中，需要多进行一次选择：既要选择特征维度，也要选择该特征维度的切分点。同样地，结点中需要存储切分点信息。
