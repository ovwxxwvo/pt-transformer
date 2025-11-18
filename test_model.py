
import torch
import torch.nn as nn
# 假设你的nn_ext模块和Transformer在同一目录，若路径不同需调整
from transformer.model import Transformer  # 替换为你的Transformer所在文件路径

# -------------------------- 1. 测试配置（可自定义）--------------------------
src_vocab_size = 1000    # 源语言词汇表大小
tgt_vocab_size = 1000    # 目标语言词汇表大小
pad_id = 0               # 填充标记ID（与模型一致）
batch_size = 2           # 批次大小
src_seq_len = 8          # 源序列长度（不等长模拟）
tgt_seq_len = 6          # 目标序列长度（不等长模拟）
d_model = 128            # 模型维度（缩小维度加快测试）
n_heads = 2              # 注意力头数
enc_n_layers = 2         # 编码器层数
dec_n_layers = 2         # 解码器层数
d_ff = 512               # 前馈网络维度

# -------------------------- 2. 生成测试数据（模拟真实输入）--------------------------
def generate_test_data(batch_size, max_src_len, max_tgt_len, vocab_size, pad_id):
    """生成不等长序列，补0对齐为等长张量（模拟真实数据预处理）"""
    # 生成源序列（batch_size个不等长序列，长度3~max_src_len）
    src_seqs = []
    for _ in range(batch_size):
        seq_len = torch.randint(3, max_src_len+1, (1,)).item()
        seq = torch.randint(1, vocab_size, (seq_len,))  # 1~vocab_size（避免pad_id=0）
        # 补0对齐到max_src_len
        padded_seq = torch.cat([seq, torch.full((max_src_len - seq_len,), pad_id, dtype=torch.long)])
        src_seqs.append(padded_seq)
    src_tensor = torch.stack(src_seqs)  # (batch_size, max_src_len)

    # 生成目标序列（batch_size个不等长序列，长度2~max_tgt_len）
    tgt_seqs = []
    for _ in range(batch_size):
        seq_len = torch.randint(2, max_tgt_len+1, (1,)).item()
        seq = torch.randint(1, vocab_size, (seq_len,))
        padded_seq = torch.cat([seq, torch.full((max_tgt_len - seq_len,), pad_id, dtype=torch.long)])
        tgt_seqs.append(padded_seq)
    tgt_tensor = torch.stack(tgt_seqs)  # (batch_size, max_tgt_len)

    return src_tensor, tgt_tensor

# 生成测试数据
src_input, tgt_input = generate_test_data(
    batch_size=batch_size,
    max_src_len=src_seq_len,
    max_tgt_len=tgt_seq_len,
    vocab_size=src_vocab_size,
    pad_id=pad_id
)

print("="*60)
print("测试数据生成完成")
print(f"源序列形状：{src_input.shape} (batch_size, max_src_len)")
print(f"源序列数据：\n{src_input}")
print(f"\n目标序列形状：{tgt_input.shape} (batch_size, max_tgt_len)")
print(f"目标序列数据：\n{tgt_input}")
print("="*60)

# -------------------------- 3. 实例化Transformer模型--------------------------
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    pad_id=pad_id,
    d_model=d_model,
    n_heads=n_heads,
    enc_n_layers=enc_n_layers,
    dec_n_layers=dec_n_layers,
    d_ff=d_ff,
    src_dropout=0.1,
    tgt_dropout=0.1,
    enc_need_weight=True,  # 测试时返回注意力权重
    dec_need_weight=True
)

# 设为测试模式（禁用dropout）
model.eval()

print("\n模型实例化完成，参数概况：")
print(f"模型总参数量：{sum(p.numel() for p in model.parameters()):,}")
print("="*60)

# -------------------------- 4. 前向传播测试--------------------------
with torch.no_grad():  # 禁用梯度计算，加快测试
    logits, attn_weights = model(src_input, tgt_input)

# -------------------------- 5. 结果验证（确保输出符合预期）--------------------------
print("前向传播测试完成，结果验证：")
print(f"输出Logits形状：{logits.shape} (batch_size, max_tgt_len, tgt_vocab_size)")
assert logits.shape == (batch_size, tgt_seq_len, tgt_vocab_size), "Logits形状不符合预期！"

# 验证注意力权重
print(f"\n编码器注意力权重层数：{len(attn_weights['enc'])} (与enc_n_layers一致)")
print(f"解码器注意力权重层数：{len(attn_weights['dec'])} (与dec_n_layers一致)")
assert len(attn_weights['enc']) == enc_n_layers, "编码器注意力权重层数错误！"
assert len(attn_weights['dec']) == dec_n_layers, "解码器注意力权重层数错误！"

# 打印部分结果示例
print(f"\nLogits前2个样本的前3个位置输出维度：{logits[:2, :3].shape}")
print(f"第一个编码器层的注意力权重形状：{attn_weights['enc'][0][0].shape}")
print(f"第一个解码器层的交叉注意力权重形状：{attn_weights['dec'][0][0].shape}")

print("\n" + "="*60)
print("🎉 Transformer模型测试全部通过！无报错且输出符合预期～")
print("="*60)
