# 目标：实现一个简单且高效的多头注意力机制

import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]])  # step    (x^6)

# 获取基于x2的注意力权重
query = inputs[1]  # A:第二个输入 token 用作查询向量
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

# 通过求和，权重归一化
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# 通过softmax 函数来进行归一化
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# 通过pytorch的softmax实现归一化
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# 计算上下文向量：计算方法为将每个输入向量与对应的注意力权重相乘后相加
query = inputs[1] # 2nd input token is the query
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)

# 通过for循环，计算所有输入对之间的点积
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

# 通过矩阵乘法计算点积结果
attn_scores = inputs @ inputs.T
print(attn_scores)

# 对每一行进行归一化处理
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# 通过矩阵乘法的方式来并行计算所有的上下文向量
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)