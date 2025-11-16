import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, pad_id):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

    def forward(self, x:torch.tensor):
        x_embed = self.embedding(x)
        scale   =  math.sqrt(self.d_model)
        x_embed = x_embed * scale
        return x_embed

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor):
        seq_len = x.size(1)
        x_posi = self.dropout(x + self.pe[:, :seq_len])
        return x_posi

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.expansion = nn.Linear(d_model, d_ff)
        self.reduction = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor):
        x_feed = self.reduction(self.dropout(self.activation(self.expansion(x))))
        return x_feed


class MaskGenerator():
    @staticmethod
    def gen_pad_mask(x: torch.Tensor, pad_id) -> torch.Tensor:
        pad_mask = (x == pad_id).to(x.device)
        # pad_mask = pad_mask.transpose(0, 1)
        return pad_mask

    def gen_dymanic_mask(x: torch.Tensor, pad_id) -> torch.Tensor:
        return dymanic_mask

    @staticmethod
    def gen_src_mask(x: torch.Tensor) -> torch.Tensor:
        src_mask = None
        return src_mask

    @staticmethod
    def gen_tgt_mask(x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        return tgt_mask


# class ScaleDotAttention(nn.Module):
    # def forward(self, q,k,v, mask=None):
    #     d_k = q.size(-1)
    #     attn_matmul = torch.matmul(q, k.transpose(-2, -1))
    #     attn_sqrt   = q.new_ones(1).fill_(math.sqrt(d_k))
    #     attn_score  = attn_matmul / attn_sqrt
    #     if mask is not None:
    #         attn_score = attn_score.masked_fill(mask == 0, -1e9)
    #     attn_weight = F.softmax(attn_score, dim=-1)
    #     attn_output = torch.matmul(attn_weight, v)
    #     return attn_output, attn_weight
    #

# class MultiHeadAttention(nn.Module):
    # def __init__(self, d_model, n_heads, dropout):
    #     super().__init__()
    #     self.n_heads = n_heads
    #     self.d_k = d_model // n_heads
    #     self.W_Q = nn.Linear(d_model, d_model)
    #     self.W_K = nn.Linear(d_model, d_model)
    #     self.W_V = nn.Linear(d_model, d_model)
    #     self.W_O = nn.Linear(d_model, d_model)
    #     self.attention = ScaleDotAttention()
    #     self.dropout = nn.Dropout(dropout)
    #
    # def forward(self, x_q, x_k, x_v, mask):
    #     batch_size = x_q.size(0)
    #     q = self.W_Q(x_q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
    #     k = self.W_K(x_k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
    #     v = self.W_V(x_v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
    #     attn_output, attn_weight = self.attention(q,k,v, mask)
    #     attn_output = attn_output.transpose(1, 2).contiguous()
    #     attn_output = attn_output.view(batch_size, -1, self.n_heads * self.d_k)
    #     x_attn = self.dropout(self.W_O(attn_output))
    #     return x_attn, attn_weight
    #


