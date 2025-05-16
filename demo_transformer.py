import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2,3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2) # n x b x l x dv

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting

        output, attn = ScaledDotProductAttention()(q, k, v, mask=mask)

        # Concatenate heads and final projection
        output = output.transpose(1, 2).contiguous().view(sz_b, -1, n_head * d_v)
        output = self.fc(output)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        attn_output, enc_slf_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        attn_output *= non_pad_mask

        enc_output = enc_input + self.dropout(attn_output)
        enc_output = self.norm1(enc_output)

        ff_output = self.pos_ffn(enc_output)
        ff_output *= non_pad_mask

        enc_output = enc_output + self.dropout(ff_output)
        enc_output = self.norm2(enc_output)

        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, dec_input, enc_output,
            non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):

        # Self Attention
        attn_output, dec_slf_attn = self.self_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        attn_output *= non_pad_mask
        dec_output = dec_input + self.dropout(attn_output)
        dec_output = self.norm1(dec_output)

        # Encoder-Decoder Attention
        attn_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        attn_output *= non_pad_mask
        dec_output = dec_output + self.dropout(attn_output)
        dec_output = self.norm2(dec_output)

        # Position-wise Feed-Forward
        ff_output = self.pos_ffn(dec_output)
        ff_output *= non_pad_mask
        dec_output = dec_output + self.dropout(ff_output)
        dec_output = self.norm3(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn

class Transformer(nn.Module):
    def __init__(
            self, n_src_vocab, n_trg_vocab,
            d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            dropout=0.1, n_position=200):

        super().__init__()
        self.src_word_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=0)
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_model, padding_idx=0)

        self.position_enc = PositionalEncoding(d_model, n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, pad_idx=0)
        trg_mask = get_pad_mask(trg_seq, pad_idx=0) & get_subsequent_mask(trg_seq)

        enc_output = self.src_word_emb(src_seq) * self.position_enc(src_seq.size(1))
        enc_output = self.dropout(enc_output)

        for enc_layer in self.encoder:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=src_mask)

        dec_output = self.trg_word_emb(trg_seq) * self.position_enc(trg_seq.size(1))
        dec_output = self.dropout(dec_output)

        for dec_layer in self.decoder:
            dec_output, _, _ = dec_layer(
                dec_output, enc_output,
                non_pad_mask=trg_mask)

        seq_logit = self.tgt_word_prj(dec_output)
        return seq_logit

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

model = Transformer(
    n_src_vocab=10000,
    n_trg_vocab=10000,
    d_model=512,
    d_inner=2048,
    n_layers=6,
    n_head=8,
    d_k=64,
    d_v=64
)

src = torch.randint(0, 10000, (32, 30))
trg = torch.randint(0, 10000, (32, 35))
logits = model(src, trg)
print(logits.shape)  # torch.Size([32, 35, 10000])