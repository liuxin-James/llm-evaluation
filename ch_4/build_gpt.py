from ch_4.build_transformer import TransformerBlock, LayerNorm
import torch
import torch.nn as nn
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)

'''
使用 numel() 方法（即 number of elements 的缩写），可以统计模型中参数张量的总参数量
'''
total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")

'''
之前提到 GPT 模型的参数量为 1.24 亿，但代码输出的实际参数量却是 1.63 亿
这是因为 GPT-2 架构中使用了一种称为‘权重共享’的概念，这意味着 GPT-2 架构将 token 嵌入层的权重复用于输出层
'''
# print("Token embedding layer shape:", model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)

'''
根据权重共享原则，我们可以从 GPT-2 模型的总参数量中去除输出层的参数量
模型现在的参数量为 1.24 亿，与 GPT-2 原始模型的规模一致
'''
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

'''
计算 GPTModel 对象中 1.63 亿参数所需的内存
'''
total_size_bytes = total_params * 4                 #A
total_size_mb = total_size_bytes / (1024 * 1024)    #B
# print(f"Total size of the model: {total_size_mb:.2f} MB")

#A 计算参数总大小（假设每个参数为 float32 类型，占用 4 字节）
#B 转换为 MB

# A function for the GPT model to generate text
def generate_text_simple(model, idx, max_new_tokens, context_size): #A
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                           #B
        with torch.no_grad():
           logits = model(idx_cond)

        logits = logits[:, -1, :]                                   #C
        probas = torch.softmax(logits, dim=-1)                      #D
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)       #E
        idx = torch.cat((idx, idx_next), dim=1)                     #F

    return idx


#A idx 是当前上下文中索引的数组，形状为 (batch, n_tokens)
#B 若上下文长度超出支持范围，则进行裁剪。例如，若模型仅支持 5 个 token，而上下文长度为 10，仅使用最后 5 个 token 作为上下文
#C 仅关注最后一个时间步，将形状从 (batch, n_token, vocab_size) 转换为 (batch, vocab_size)
#D probas 的形状为 (batch, vocab_size)
#E idx_next 的形状为 (batch, 1)
#F 将采样的索引追加到当前序列中，此时 idx 的形状为 (batch, n_tokens+1)

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
# print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # 添加批次维度
# print("encoded_tensor.shape:", encoded_tensor.shape)

# 将模型置于 .eval() 模式，禁用训练时使用的随机组件（如 dropout）
model.eval()             # 禁用 dropout，因为当前不是在训练模型
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=7,
    context_size=GPT_CONFIG_124M["context_length"]
)
# print("Output:", out)
# print("Output length:", len(out[0]))

# 对输出进行解码
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)