# 实现带有可训练权重的自注意力机制

import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]])  # step    (x^6)

x_2 = inputs[1]  # A
d_in = inputs.shape[1]  # B
d_out = 2  # C

# A 第二个输入元素
# B 输入维度， d_in=3
# C 输出维度， d_out=2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# 通过矩阵乘法获取所有元素的key和value向量
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# 计算注意力得分ω22
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# 通过矩阵乘法将其应用到所有注意力得分的计算
attn_scores_2 = query_2 @ keys.T  # All attention scores for given query
print(attn_scores_2)

# 通过softmax 函数来计算注意力权重
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print(attn_weights_2)

'''
缩放点积注意力机制的原理:
对嵌入维度大小进行归一化的原因是为了避免出现小梯度，从而提高训练性能
通过嵌入维度的平方根进行缩放，正是自注意力机制被称为‘缩放点积注意力’的原因

在自注意力机制中，查询向量（Query）与键向量（Key）之间的点积用于计算注意力权重。然而，当嵌入维度（embedding dimension）较大时，点积的结果可能会非常大，对计算有影响：
1。Softmax函数的特性：在计算注意力权重时，点积结果会通过Softmax函数转换为概率分布。而Softmax函数对输入值的差异非常敏感，当输入值较大时，Softmax的输出会趋近于0或1，表现得类似于阶跃函数（step function）
2。梯度消失问题：当Softmax的输出接近0或1时，其梯度会非常小，接近于零（可以通过3.3.1小节中提到的Softmax公式推断）。这意味着在反向传播过程中，梯度更新幅度会很小，导致模型学习速度减慢，甚至训练停滞
'''
# 通过值向量的加权和来计算上下文向量
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# A compact self-attention class
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

'''
我们可以通过使用 PyTorch 的 nn.Linear 层来进一步改进 SelfAttention_v1 的实现。当禁用偏置单元时，nn.Linear 层可以有效地执行矩阵乘法。
此外，使用 nn.Linear 替代手动实现的 nn.Parameter(torch.rand(...)) 的一个显著优势在于，nn.Linear 具有优化的权重初始化方案，从而有助于实现更稳定和更高效的模型训练。
'''
# A self-attention class using PyTorch's Linear layers
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

