'''
因果注意力（也称为掩蔽注意力）是一种特殊的自注意力形式。它限制模型在处理任何给定的 token 时，只能考虑序列中的前一个和当前输入，而不能看到后续的内容。
这与标准的自注意力机制形成对比，后者允许模型同时访问整个输入序列。
在计算注意力分数时，因果注意力机制确保模型只考虑当前 token 或之前的 token
'''
import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]])  # step    (x^6)


# A self-attention class using PyTorch's Linear layers
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


d_in = inputs.shape[1]  # 3
d_out = 2  # 2
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)

# 生成掩码矩阵，将对角线右上方进行掩码处理
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
masked_simple = attn_weights * mask_simple
print(masked_simple)

# 归一化处理，通过将每一行中的每个元素除以该行的总和来实现
row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

'''
通过创建一个对角线以上全为 1 的掩码，然后将这些 1 替换为负无穷大（-inf）值，从而实现这种更高效的掩码技巧
'''
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)

'''
使用 dropout 遮掩额外的注意力权重: Dropout 在深度学习中是一种技术，即在训练过程中随机忽略一些隐藏层单元，实际上将它们“丢弃”
在 Transformer 架构中（包括 GPT 等模型），注意力机制中的 Dropout 通常应用于两个特定区域：计算注意力得分之后，或将注意力权重应用于 value 向量之后
'''
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)                                   #A
example = torch.ones(6, 6)                                        #B
print(dropout(example))

#A 我们使用的dropout率为0.5
#B 创建一个由1组成的矩阵

'''
当对注意力权重矩阵应用 50% 的 dropout 时，矩阵中一半的元素会被随机设置为零。
为了补偿有效元素的减少，矩阵中剩余元素的值会被放大 1/0.5 = 2 倍。
这个缩放操作至关重要，可以在训练和推理阶段保持注意力机制的整体权重平衡，确保注意力机制在这两个阶段的平均影响保持一致
'''
torch.manual_seed(123)
print(dropout(attn_weights))

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)      # 2个输入，每个输入有6个token，每个token的嵌入维度为3

# A compact causal attention class
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)                        #A
        self.register_buffer(
           'mask',
           torch.triu(torch.ones(context_length, context_length),
           diagonal=1)
        )                                                         #B

    def forward(self, x):
        b, num_tokens, d_in = x.shape                             #C
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)              #C
        attn_scores.masked_fill_(                                 #D
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


#A 与之前的 SelfAttention_v1 类相比，我们添加了一个 dropout 层
#B register_buffer 调用也是新添加的内容（后续内容会提供更多相关信息）
#C 我们交换第 1 和第 2 个维度，同时保持批次维度在第1个位置（索引0）
#D 在 PyTorch 中，带有下划线后缀的操作会在原有内存空间执行，直接修改变量本身，从而避免不必要的内存拷贝

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
print(context_vecs)