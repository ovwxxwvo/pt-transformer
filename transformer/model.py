import torch
import torch.nn as nn
from . import nn_ext as nn_


class EncodeLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, need_weights):
        super().__init__()
        batch_first = True
        self.need_weights = need_weights
        self.unmask_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=batch_first, dropout=dropout)
        self.feedforward = nn_.FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.feed_norm = nn.LayerNorm(d_model)

    def forward(self, x_src, src_attn_mask, src_pad_mask):
        x = x_src
        need_weights = self.need_weights
        x_attn, unmask_attn_weight = self.unmask_attn(x, x, x,
            attn_mask=src_attn_mask,
            key_padding_mask=src_pad_mask,
            need_weights=need_weights,
            is_causal=False,
            )
        x      = self.attn_norm(x + x_attn)
        x_feed = self.feedforward(x)
        x      = self.feed_norm(x + x_feed)
        attn_weight = (unmask_attn_weight,)
        return x, attn_weight

class DecodeLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, need_weights):
        super().__init__()
        batch_first = True
        self.need_weights = need_weights
        self.mask_attn   = nn.MultiheadAttention(d_model, n_heads, batch_first=batch_first, dropout=dropout)
        self.cross_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=batch_first, dropout=dropout)
        self.feedforward = nn_.FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.m_attn_norm = nn.LayerNorm(d_model)
        self.c_attn_norm = nn.LayerNorm(d_model)
        self.feed_norm   = nn.LayerNorm(d_model)

    def forward(self, x_tgt, x_src, tgt_attn_mask, tgt_pad_mask, src_attn_mask, src_pad_mask):
        x = x_tgt
        need_weights = self.need_weights
        x_attn, mask_attn_weight = self.mask_attn(x, x, x,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_pad_mask,
            need_weights=need_weights,
            is_causal=True,
            )
        x      = self.m_attn_norm(x + x_attn)
        x_attn, cross_attn_weight = self.cross_attn(x, x_src, x_src,
            attn_mask=src_attn_mask,
            key_padding_mask=src_pad_mask,
            need_weights=need_weights,
            is_causal=False,
            )
        x      = self.c_attn_norm(x + x_attn)
        x_feed = self.feedforward(x)
        x      = self.feed_norm(x + x_feed)
        attn_weight = (cross_attn_weight, mask_attn_weight,)
        return x, attn_weight

class InputLayer(nn.Module):
    def __init__(self, d_model, vocab_size, pad_id, dropout):
        super().__init__()
        self.embed    = nn_.TokenEmbedding(d_model, vocab_size, pad_id)
        self.position = nn_.PositionalEncoding(d_model, dropout=dropout)

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


class Transformer(nn.Module):
    def __init__(self,
        pad_id,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        d_ff=1024,
        n_heads=4,
        enc_n_layers=4,
        dec_n_layers=4,
        src_dropout=0.1,
        tgt_dropout=0.1,
        enc_need_weight=False,
        dec_need_weight=False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.src_input = InputLayer(d_model, src_vocab_size, pad_id, src_dropout)
        self.tgt_input = InputLayer(d_model, tgt_vocab_size, pad_id, tgt_dropout)
        self.output    = OutputLayer(d_model, tgt_vocab_size)
        self.encodes = nn.ModuleList([EncodeLayer(d_model, n_heads, d_ff, src_dropout, enc_need_weight) for _ in range(enc_n_layers)])
        self.decodes = nn.ModuleList([DecodeLayer(d_model, n_heads, d_ff, tgt_dropout, dec_need_weight) for _ in range(dec_n_layers)])
        # print("=" * 40)
        # print(f"Transformer Model Initialized.")
        # print("=" * 40)

    def forward(self, x_src, x_tgt):
        pad_id = self.pad_id
        src_pad_mask  = nn_.MaskGenerator().gen_pad_mask(x_src, pad_id)
        tgt_pad_mask  = nn_.MaskGenerator().gen_pad_mask(x_tgt, pad_id)
        src_attn_mask = nn_.MaskGenerator().gen_src_mask(x_src)
        tgt_attn_mask = nn_.MaskGenerator().gen_tgt_mask(x_tgt)

        x_src = self.src_input(x_src)
        x_tgt = self.tgt_input(x_tgt)
        enc_attn_weights = []
        dec_attn_weights = []

        for enc in self.encodes:
            x_src, enc_attn_weight = enc(x_src, src_attn_mask, src_pad_mask)
            enc_attn_weights.append(enc_attn_weight)
        for dec in self.decodes:
            x_tgt, dec_attn_weight = dec(x_tgt, x_src, tgt_attn_mask, tgt_pad_mask, src_attn_mask, src_pad_mask)
            dec_attn_weights.append(dec_attn_weight)

        logits = self.output(x_tgt)
        attn_weights = {"enc":tuple(enc_attn_weights), "dec":tuple(dec_attn_weights)}
        return logits, attn_weights


