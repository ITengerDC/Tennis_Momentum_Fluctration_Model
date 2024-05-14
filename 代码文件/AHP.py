import numpy as np

# 定义成对比较矩阵
pairwise_comparison_matrix = np.array([
    [1, 2, 4, 5],
    [1/2, 1, 3, 4],
    [1/4, 1/3, 1, 2],
    [1/5, 1/4, 1/2, 1]
])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(pairwise_comparison_matrix)

# 找到最大特征值及其对应的特征向量
max_index = np.argmax(eigenvalues.real)
max_eigenvalue = eigenvalues[max_index].real
principal_eigenvector = eigenvectors[:, max_index].real

# 计算权重
weights = principal_eigenvector / np.sum(principal_eigenvector)

# 计算一致性指标 (CI)
n = pairwise_comparison_matrix.shape[0]  # 矩阵维度
CI = (max_eigenvalue - n) / (n - 1)

# 随机一致性指数 (RI) 对应不同维度的值
RI_values = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
RI = RI_values[n]

# 计算一致性比率 (CR)
CR = CI / RI

print("Weights:", weights)
print("CR:", CR)

# 检查一致性比率是否小于0.1，以判断成对比较矩阵的一致性是否可接受
if CR < 0.1:
    print("The pairwise comparison matrix has acceptable consistency.")
else:
    print("The pairwise comparison matrix may need to be re-evaluated due to poor consistency.")
