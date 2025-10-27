import numpy as np
from scipy.stats import ttest_ind


# group1 和 group2 可以是两个模型的准确率、AUC等结果列表

group1 = [1.2, 1.3, 1.1, 1.0, 1.2]  # 随机种子或交叉验证
group2 = [2.0, 2.1, 2.2, 2.1, 2.0]

# 独立样本t检验
t_stat, p_value = ttest_ind(group1, group2)

print(f"t值: {t_stat}, p值: {p_value}")
if p_value < 0.01:
    print("两组数据差异极显著 (p < 0.01)")
elif p_value < 0.05:
    print("两组数据有显著性差异 (0.01 ≤ p < 0.05)")
else:
    print("两组数据无显著性差异 (p ≥ 0.05)")
