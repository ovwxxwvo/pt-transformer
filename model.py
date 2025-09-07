import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x:torch.tensor):
        x_embed = self.embedding(x)
        scale =  math.sqrt(self.d_model)
        x_embed = x_embed * scale
        return x_embed

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor):
        batch_size, seq_len, _ = x.shape
        device = x.device
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x_posi = self.dropout(x + pe.unsqueeze(0))
        return x_posi

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.expansion = nn.Linear(d_model, d_ff)
        self.reduction = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor):
        x_feed = self.reduction(self.dropout(self.activation(self.expansion(x))))
        return x_feed

class ScaleDotAttention(nn.Module):
    def forward(self, q,k,v, mask=None):
        d_k = q.size(-1)
        attn_matmul = torch.matmul(q, k.transpose(-2, -1))
        attn_sqrt   = q.new_ones(1).fill_(math.sqrt(d_k))
        attn_score  = attn_matmul / attn_sqrt
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_output = torch.matmul(attn_weight, v)
        return attn_output, attn_weight

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attention = ScaleDotAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, mask):
        batch_size = x_q.size(0)
        q = self.W_Q(x_q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x_k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(x_v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weight = self.attention(q,k,v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.n_heads * self.d_k)
        x_attn = self.dropout(self.W_O(attn_output))
        return x_attn, attn_weight


class EncodeLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.unmask_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feedforward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.feed_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x_attn, _ = self.unmask_attn(x, x, x, mask)
        x         = self.attn_norm(x + x_attn)
        x_feed    = self.feedforward(x)
        x         = self.feed_norm(x + x_feed)
        return x

class DecodeLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.feedforward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.m_attn_norm = nn.LayerNorm(d_model)
        self.c_attn_norm = nn.LayerNorm(d_model)
        self.feed_norm   = nn.LayerNorm(d_model)

    def forward(self, x, x_encode, tgt_mask, src_mask):
        x_m_attn, _ = self.masked_attn(x, x, x, tgt_mask)
        x           = self.m_attn_norm(x + x_m_attn)
        x_c_attn, _ = self.cross_attn(x, x_encode, x_encode, src_mask)
        x           = self.c_attn_norm(x + x_c_attn)
        x_feed      = self.feedforward(x)
        x           = self.feed_norm(x + x_feed)
        return x

class InputLayer(nn.Module):
    def __init__(self, d_model, vocab_size, dropout):
        super().__init__()
        self.embed    = TokenEmbedding(d_model, vocab_size)
        self.position = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        x = self.embed(x)
        x = self.position(x)
        return x

class OutputLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.full_conn = nn.Linear(d_model, vocab_size)
        # self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.full_conn(x)
        # x = self.softmax(x)
        return x


class MaskGenerator():

    @staticmethod
    def gen_pad_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        pad_mask = (seq == pad_token).unsqueeze(1).unsqueeze(2)
        return pad_mask

    @staticmethod
    def gen_tgt_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        seq_len = seq.size(1)
        triu_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()
        pad_mask = MaskGenerator.gen_pad_mask(seq, pad_token).expand(-1, 1, seq_len, -1)
        tgt_mask = torch.logical_or(pad_mask, triu_mask)
        return tgt_mask


class Transformer(nn.Module):
    def __init__(self,
                 d_model=512,
                 enc_n_heads=4,
                 dec_n_heads=4,
                 enc_n_layers=4,
                 dec_n_layers=4,
                 enc_d_ff=2048,
                 dec_d_ff=2048,
                 enc_dropout=0.1,
                 dec_dropout=0.1,
                 src_vocab_size=None,
                 tgt_vocab_size=None,
                 ):
        super().__init__()
        assert src_vocab_size is not None and src_vocab_size > 0, 'vocab_size must great than 0'
        assert tgt_vocab_size is not None and tgt_vocab_size > 0, 'vocab_size must great than 0'
        self.src_input = InputLayer(d_model, src_vocab_size, enc_dropout)
        self.tgt_input = InputLayer(d_model, tgt_vocab_size, dec_dropout)
        self.output = OutputLayer(d_model, tgt_vocab_size)
        self.encodes = nn.ModuleList([EncodeLayer(d_model, enc_n_heads, enc_d_ff, enc_dropout) for _ in range(enc_n_layers)])
        self.decodes = nn.ModuleList([DecodeLayer(d_model, dec_n_heads, dec_d_ff, dec_dropout) for _ in range(dec_n_layers)])
        print("=" * 40)
        print(f"Transformer Model Structure Initialized.")
        print("=" * 40)

    def forward(self, x_src, x_tgt):
        x_enc = self.src_input(x_src)
        src_mask = MaskGenerator().gen_pad_mask(x_src)
        tgt_mask = MaskGenerator().gen_tgt_mask(x_tgt)
        for enc in self.encodes:
            x_enc = enc(x_enc, src_mask)
        x_dec = self.tgt_input(x_tgt)
        for dec in self.decodes:
            x_dec = dec(x_dec, x_enc, tgt_mask, src_mask)
        logits = self.output(x_dec)
        return logits


