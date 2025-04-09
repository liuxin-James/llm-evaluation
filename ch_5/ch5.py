import torch
import tiktoken
from ch_4.build_gpt import GPTModel, generate_text_simple
from ch_2.dataloader import create_dataloader_v1

'''
1.1：使用 GPT 生成文本
'''
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # 我们将上下文长度从1024个token缩短到256个token
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,  # 将 dropout 设置为 0 是一种常见的做法
    "qkv_bias": False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


# Utility functions for text to token ID conversion
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

'''
1.2：计算文本生成损失
'''
inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                       [40, 1107, 588]])  # "I really like"]
# Matching these inputs, the `targets` contain the token IDs we aim for the model to produce:
targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                        [107, 588, 11311]])  # " really like chocolate"]

with torch.no_grad():  # 禁用梯度跟踪，因为我们尚未进行训练
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)  # Probability of each token in vocabulary
print(probas.shape)

# 通过对概率得分应用 argmax 函数，可以得到对应的 token ID
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

# 将token ID 转换回文本
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# When we decode these tokens, we find that these output tokens are quite different from the target tokens we want the model to generate:
# Targets batch 1: effort moves you
# Outputs batch 1: Armed heNetflix

# 对于这两个输入文本，我们可以通过以下代码打印与目标 token 对应的初始 softmax 概率得分
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

'''
如何最大化目标 token 的 softmax 概率值？整体思路是通过更新模型权重，使模型在生成目标 token 时输出更高的概率值。
通过反向传播：需要一个损失函数，该函数用于计算模型预测输出与实际目标输出之间的差异（此处指与目标 token ID 对应的概率）。这个损失函数用于衡量模型预测与目标值的偏差程度

在机器学习，特别是 PyTorch 等框架中，cross_entropy 函数用于计算离散输出的损失，与模型生成的 token 概率下的目标 token 的负平均对数概率类似
'''
# 对这些目标token的概率得分取对数
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

# 接下来，通过计算平均值将这些对数概率合并为一个评分
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

# 在深度学习中，常见做法并不是直接将平均对数概率推向 0，而是通过将负平均对数概率降低至 0 来实现
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

# 在 PyTorch 中，cross_entropy 函数能够自动完成所有上面的步骤计算损失值
# 在 PyTorch 中使用交叉熵损失函数时，我们需要将这些张量展平，以便在批量维度上进行合并
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

'''
1.3：计算训练集和验证集的损失
'''
file_path = "D://codes//llm-evaluation//ch_5//the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# 首先定义一个 train_ratio，用于将 90% 的数据用于训练，剩余 10% 用于在训练期间进行模型评估
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 复用第 2 章中的 create_dataloader_v1 代码来创建相应的数据加载器（训练数据集和测试数据集）
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


# 输入数据（x）和目标数据（y）的形状相同（即批次大小 × 每批的 token 数量），因为目标数据是将输入数据整体向后偏移一个位置得到的
# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)
#
# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)


# 实现一个工具函数calc_loss_batch，用于计算由训练和验证加载器返回的批量数据的交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据传输到指定设备（如 GPU），使数据能够在 GPU 上处理。
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


# calc_loss_loader 将用于计算指定数据加载器中的指定数据批次的损失
# Function to compute the training and validation loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  # A
    else:
        num_batches = min(num_batches, len(data_loader))  # B
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # C
        else:
            break
    return total_loss / num_batches  # D


# A 如果没有指定批次数，将自动遍历所有批次
# B 若批次数超过数据加载器的总批次数，则减少批次数使其与数据加载器的批次数相匹配
# C 每个批次的损失求和
# D 对所有批次的损失取平均值

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # A
model.to(device)
with torch.no_grad():  # B
    train_loss = calc_loss_loader(train_loader, model, device)  # C
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# A 如果你的设备配备了支持 CUDA 的 GPU，LLM 将自动在 GPU 上进行训练，无需更改代码
# B 因为当前不在训练，为提高效率，关闭梯度跟踪
# C 通过 device 设置确保数据与 LLM 模型加载到同一设备上

'''
2：预训练LLM
'''


# the main function for pretraining LLMs
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []  # A
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):  # B
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # C
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # D
            optimizer.step()  # E
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:  # F
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"------ Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(  # G
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


# A 初始化用于记录损失和已处理 token 数量的列表
# B 开始主训练循环
# C 重置上一批次的损失梯度
# D 计算损失梯度
# E 使用损失梯度更新模型权重
# F 可选的评估步骤
# G 每个 epoch 结束后打印示例文本

# evaluate_model 函数会在训练集和验证集上计算损失，同时确保模型处于评估模式，并在计算损失时禁用梯度跟踪和 dropout
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # A
    with torch.no_grad():  # B
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# A 评估阶段禁用 dropout，以确保结果稳定、可复现
# B 禁用梯度跟踪，减少计算开销

# generate_and_print_sample text函数则通过生成的实际文本示例，帮助我们在训练过程中判断模型的能力
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


'''
实例化一个GPT model，开始训练
'''
# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)  # A
# num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=1,
#     start_context="Every effort moves you", tokenizer=tokenizer
# )

# A .parameters() 方法返回模型的所有可训练权重参数

# 先创建一个简单的图表，将训练集和验证集损失并排展示
import matplotlib.pyplot as plt


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()  # A
    ax2.plot(tokens_seen, train_losses, alpha=0)  # B
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# A 创建与 y 轴共用的第二个 x 轴
# B 用于对齐刻度的隐藏图形

'''
在一个较小的数据集上进行多轮训练，容易产生过拟合的现象的原因：
1. 模型容量与数据集大小的匹配问题
2. 多轮训练导致对数据集细节的过度学习
3. 数据集的多样性不足
4. 过拟合与模型泛化能力的矛盾
'''

'''
3: 通过解码策略控制生成结果的随机性
'''
# LLM 的文本生成策略，以减少训练数据的记忆倾向，提升 LLM 生成文本的原创性
model.to("cpu")
model.eval()

'''
3.1：Temperature scaling（一种在生成下一个词时加入概率选择的技术）

为了实现概率采样过程，现在可以用 PyTorch 中的 multinomial 函数代替 argmax
'''
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# 定义一个简单的assign工具函数，用于检查两个张量或数组（左侧和右侧）的维度或形状是否一致，并将右侧张量作为可训练的 PyTorch 参数返回
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# Loading OpenAI weights into our GPT model code
import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])               #A
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):                                       #B
        q_w, k_w, v_w = np.split(                                                #C
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])                   #D


#A 将模型的位置嵌入和token 嵌入的权重设置为 params 中指定的值
#B 遍历模型中的每个 Transformer 模块
#C 使用 np.split 函数将注意力和偏置权重分为三等份，分别用于查询、键和值组件
#D OpenAI 的原始 GPT-2 模型在输出层中复用了 token 嵌入的权重，以减少参数总量，这一概念称为权重共享


