from ch_4.build_transformer import TransformerBlock

'''
Parameters in the feed forward versus attention module
'''
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

block = TransformerBlock(GPT_CONFIG_124M)
print(block)
total_params = sum(p.numel() for p in block.ff.parameters())
print(f"Total number of parameters in feed forward module: {total_params:,}")

total_params = sum(p.numel() for p in block.att.parameters())
print(f"Total number of parameters in attention module: {total_params:,}")

# Total number of parameters in feed forward module: 4,722,432
# Total number of parameters in attention module: 2,360,064

'''
参数计算范例：

Feed forward module:
1st Linear layer: 768 inputs × 4×768 outputs + 4×768 bias units = 2,362,368
2nd Linear layer: 4×768 inputs × 768 outputs + 768 bias units = 2,360,064
Total: 1st Linear layer + 2nd Linear layer = 2,362,368 + 2,360,064 = 4,722,432

Attention module:
W_query: 768 inputs × 768 outputs = 589,824
W_key: 768 inputs × 768 outputs = 589,824
W_value: 768 inputs × 768 outputs = 589,824
out_proj: 768 inputs × 768 outputs + 768 bias units = 590,592
Total: W_query + W_key + W_value + out_proj = 3×589,824 + 590,592 = 2,360,064
'''