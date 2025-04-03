# 构建一个用于生成文本的GPT模型

'''
vocab_size指的是第 2 章中 BPE 分词器使用的 50,257 个词汇的词表大小。
context_length表示模型所能处理的最大输入 token 数（在第 2 章介绍位置嵌入时讨论过）。
emb_dim表示嵌入维度，将每个 token 转换为 768 维的向量。
n_layers指定模型中 Transformer 模块的层数，后续章节将对此详解。
drop_rate表示 dropout 机制的强度（例如，0.1 表示丢弃 10% 的隐藏单元），用于防止过拟合，具体内容请回顾第 3 章。
`qkv_bia 参数决定是否在多头注意力的查询、键和值的线性层中加入偏置向量。
'''
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}

# A placeholder GPT model architecture class：DummyGPTModel 的 GPT 占位架构
import torch
import torch.nn as nn
import tiktoken


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # token 嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # dropout
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # A
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])  # B
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )  # 线性输出层（out_head）

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):  # C
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):  # D
        return x


# 最终的层归一化
class DummyLayerNorm(nn.Module):  # E
    def __init__(self, normalized_shape, eps=1e-5):  # F
        super().__init__()

    def forward(self, x):
        return x


# A 为 TransformerBlock 设置占位符
# B 为 LayerNorm 设置占位符
# C 一个简单的占位类，后续将被真正的 TransformerBlock 替换
# D 该模块无实际操作，仅原样返回输入
# E 一个简单的占位类，后续将被真正的 DummyLayerNorm 替换
# F 此处的参数仅用于模拟LayerNorm接口

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

'''
初始化一个拥有 1.24 亿参数的 DummyGPTModel 模型实例，并将分词后的数据批量输入到模型中
'''
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

'''
归一化的核心思想是将神经网络层的激活（输出）调整为均值为 0，方差为 1（即单位方差）。这种调整可以加速权重的收敛速度，确保训练过程的一致性和稳定性。
实现的神经网络层包含一个线性层，后接一个非线性激活函数 ReLU，这是神经网络中的标准激活函数: 其中ReLU作用是将负值设为 0，确保输出层中没有负值
'''
torch.manual_seed(123)
batch_example = torch.randn(2, 5)  # A
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

# A 创建2个训练样本，每个样本有5个维度（特征）

# 对这些输出应用层归一化之前，我们先查看其均值和方差
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)


# A layer normalization class
'''
scale 和 shift 是两个可训练参数（与输入具有相同的维度）。大语言模型（LLM）在训练中会自动调整这些参数，以改善模型在训练任务上的性能
'''
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

'''
层归一化与批量归一化的区别
1. 与在数量维度上进行归一化的批量归一化不同，层归一化是在特征维度上进行归一化
2. LLM 通常需要大量计算资源，而可用的硬件资源或特定的使用场景可能会限制训练或推理过程中的批量大小。
由于层归一化对每个输入的处理不依赖批量大小，因此在这些场景下提供了更高的灵活性和稳定性。这对于分布式训练或资源受限的环境中部署模型尤其有利。
'''

# An implementation of the GELU activation function : GELU（高斯误差线性单元）和 SwiGLU（Swish 门控线性单元）
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

'''
更直观的观察 GELU 函数的形状，并与 ReLU 函数进行对比
'''
import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)                                          #A
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
# plt.show()
#A 在 -3 到 3 的范围内生成 100 个样本数据点

# A feed forward neural network module
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)          #A
out = ffn(x)
print(out.shape)
#A 创建一个 batch 大小为 2 的示例输入

'''
shortcut connection
快捷连接最初是在计算机视觉中的深度网络（尤其是残差网络）提出的，用于缓解梯度消失问题。
梯度消失是指在训练中指导权重更新的梯度在反向传播过程中逐渐减小，导致早期层（靠近输入端的网络层）难以有效训练
'''
# A neural network to illustrate shortcut connections
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            # Implement 5 layers
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

