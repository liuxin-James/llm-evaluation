import torch

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6 # 词汇表为6
output_dim = 3 # 嵌入向量大小为3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)