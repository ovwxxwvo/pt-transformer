import random
import torch
from transformer.nn_ext import MaskGenerator


def test_mask_generators():
    print("="*40)
    print("测试流程：不等长序列 → 补0对齐序列 → 分别生成Pad和Tgt的掩码")
    print("="*40)

    pad_id = 0      # 补位标记（移到最前）
    batch_size  = 4 # 行数=批次大小（一次处理的样本数）
    max_seq_len = 8 # 最大序列长度（补0后的列数）
    raw_seqs = []   # 存储不等长原始序列
    print(f"参数：pad_id={pad_id}，batch_size={batch_size}，max_seq_len={max_seq_len}")

# 第一步：生成不等长原始序列
    print("\n1. 生成不等长原始序列（列=seq_len，每行长度随机2~max_seq_len）：")
    for i in range(batch_size):
        seq_len = random.randint(2, max_seq_len)     # 每行seq_len随机（不等长）
        seq = random.sample(range(10, 100), seq_len) # 随机生成10~99的真实数据（无pad_id）
        raw_seqs.append(seq)
        print(f"  样本{i+1}（seqlen={seq_len}）：{seq}")

# 第二步：补0对齐，生成等长张量（batch_size, max_seq_len）
    padded_seqs = [seq + [pad_id]*(max_seq_len - len(seq)) for seq in raw_seqs]
    x = torch.tensor(padded_seqs)
    print(f"\n2. 补0后的等长张量（shape=[batch_size, max_seq_len]={x.shape}）：")
    print(x)

# 第三步：生成Pad Mask（标记pad_id位置）
    pad_mask = MaskGenerator.gen_pad_mask(x, pad_id)
    print(f"\n3. Pad Mask（True=pad_id={pad_id}，需掩蔽；shape=[max_seq_len, max_seq_len]={pad_mask.shape}）：")
    print(pad_mask)

# 第四步：生成Tgt Mask（掩蔽未来token，正三角）
    tgt_mask = MaskGenerator.gen_tgt_mask(x)
    print(f"\n4. Tgt Mask（True=未来token，需掩蔽；shape=[max_seq_len, max_seq_len]={tgt_mask.shape}）：")
    print(tgt_mask)

# 验证逻辑（确保功能正确）

    # 验证Pad Mask：补0的位置全为True
    for i, seq in enumerate(raw_seqs):
        assert not pad_mask[i, :len(seq)].any(), f"样本{i+1}真实数据被误掩蔽"
        assert pad_mask[i, len(seq):].all(), f"样本{i+1}补位0未被掩蔽"

    # 验证Tgt Mask：正三角逻辑
    seq_len = x.size(1)
    for i in range(seq_len):
        assert not tgt_mask[i, :i+1].any(), f"第{i+1}个token的当前/历史被误掩蔽"
        assert tgt_mask[i, i+1:].all(), f"第{i+1}个token的未来未被掩蔽"

    print("\n" + "="*40)
    print("🎉 所有流程测试通过！双掩码生成符合预期～")
    print("="*40)


if __name__ == "__main__":
    test_mask_generators()


