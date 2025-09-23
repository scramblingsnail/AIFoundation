# **朴素贝叶斯法**

## **待解决问题**
现有N个训练样本，每个样本K个特征，可能类别有C种。
给定测试样本，判断测试样本的类别。

## **解决方案：用先验概率估计后验概率**
现在的问题是，已知样本的特征数据 $X$ ，求样本的类别 $Y$ ，或者说，求概率 $P(Y|X)$ 最大的类别。

根据贝叶斯公式：

$$
P(Y|X) = \frac {P(X, Y)} {P(X)} = \frac {\sum_Y P(X|Y)P(Y)} {P(X)}
$$

我们的目标是判断 $Y$ 不同取值情况下，后验概率的相对大小。上式中分母 $P(X)$ 与该目标无关，可省略。

上式分子中的先验概率 $P(Y)$ 容易从样本数据中估计，但计算 $P(X|Y)$ 时，由于特征维度可能很多，该值不易求得。为此，朴素贝叶斯法做了一个强假设：
> 各个特征维度彼此相互独立，因此 $P(x_1, x_2, ..., x_K|Y) = P(x_1|Y)P(x_2|Y) ... P(x_K|Y)$

于是，通过从训练样本数据中估计各个类别的先验概率 $P(Y)$，以及测试样本各个特征维度取值的条件概率 $P(x_1|Y), ... P(x_K|Y)$，
即可得出后验概率 $P(Y|(x_1, x_2, ..., x_K))$ ，选取后验概率最大的类别作为分类类别即可。

## **代码**

<details>
<summary>朴素贝叶斯法</summary>
```python
import numpy as np


class Bayes:
    def __init__(self):
        self.data = self.load_data()
        self.label_pri_probs = self.collect_pri_prob()
        self.label_fval_cond_probs = self.collect_cond_prob()
        # print(self.label_pri_probs)
        # print(self.label_fval_cond_probs)

    def load_data(self):
        N, K = 1000, 5
        class_num = 10
        feature_min, feature_max = 0, 3
        f_data = np.random.randint(feature_min, feature_max, (N, K))
        labels = np.random.randint(0, class_num, (N, 1))
        data = np.concatenate((f_data, labels), axis=-1)
        return data

    def collect_pri_prob(self, ):
        # return: dict, Key -- label, Value -- prob
        all_num = self.data.shape[0]
        label_dict = dict()
        for d_idx in range(all_num):
            label = self.data[d_idx, -1]
            count = label_dict.get(label, 0)
            label_dict[label] = count + 1
        for label in label_dict.keys():
            label_dict[label] /= all_num
        return label_dict

    def collect_cond_prob(self,):
        # P(x_i|Y)
        # Return: dict, Key -- label; Value: dict (Key -- Feature; Value -- dict (Key -- val_of_this_feature, val -- count))
        # 查询：label --> feature --> f_val
        cond_prob_dict = dict()
        all_num = self.data.shape[0]
        f_num = self.data.shape[-1]
        for d_idx in range(all_num):
            label = self.data[d_idx, -1]
            label_dict = cond_prob_dict.get(label, dict())
            for f_idx in range(f_num):
                feature_dict = label_dict.get(f_idx, dict())
                f_val = self.data[d_idx][f_idx]
                count = feature_dict.get(f_val, 0)
                feature_dict[f_val] = count + 1
                label_dict[f_idx] = feature_dict
            cond_prob_dict[label] = label_dict

        for label in cond_prob_dict.keys():
            label_dict = cond_prob_dict[label]
            for f_idx in label_dict.keys():
                f_dict = label_dict[f_idx]
                count_sum = sum(list(f_dict.values()))
                for f_val in f_dict.keys():
                    f_dict[f_val] /= count_sum
        return cond_prob_dict

    def classify(self, test_vec):
        post_prob_list = []
        max_prob = 0
        predict = None

        for label in self.label_pri_probs.keys():
            pri_prob = self.label_pri_probs.get(label, 0)
            cond_prob = 1
            for f_idx in range(test_vec.shape[0]):
                test_f_val = test_vec[f_idx]
                cond_prob *= self.label_fval_cond_probs[label][f_idx].get(test_f_val, 0)

            post_prob = cond_prob * pri_prob
            # print(f"Label: {label}, pri_prob: {pri_prob} -- cond_prob: {cond_prob} -- post_prob: {post_prob} -- max_prob: {max_prob}")
            if post_prob > max_prob:
                predict = label
                max_prob = post_prob
        return predict


if __name__ == "__main__":
    bayes = Bayes()
    test_features = np.random.randint(0, 3, (100, 5))
    for d_idx in range(test_features.shape[0]):
        ll = bayes.classify(test_features[d_idx])
        print(f"No.{d_idx}: ", ll)

```

</details>