import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import random, torch
from transformer.nn_ext import MaskGenerator


def main():
    print(f"-- Test Mask Generation --")
    print("=" * 40)

    print("\nTest Flow: Variable-length Sequences → Zero-padding to Align → Generate Pad Mask & Tgt Mask Separately")

    pad_id = 0      # Padding token ID (moved to front)
    batch_size = 3  # Number of samples per batch (rows)
    max_seq_len = 6 # Maximum sequence length (columns after padding)
    raw_seqs = []   # Store variable-length original sequences
    print(f"Parameters: pad_id={pad_id}, batch_size={batch_size}, max_seq_len={max_seq_len}")

# Step 1: Generate variable-length original sequences
    print("\n1. Generate variable-length original sequences (columns=seq_len, length random from 2~max_seq_len):")
    for i in range(batch_size):
        seq_len = random.randint(2, max_seq_len)     # Random sequence length for each sample
        seq = random.sample(range(10, 100), seq_len) # Randomly generate real data (10~99, no pad_id)
        raw_seqs.append(seq)
        print(f"  Sample {i+1} (seq_len={seq_len}): {seq}")

# Step 2: Zero-padding to get fixed-length tensor (batch_size, max_seq_len)
    padded_seqs = [seq + [pad_id] * (max_seq_len - len(seq)) for seq in raw_seqs]
    x = torch.tensor(padded_seqs)
    print(f"\n2. Fixed-length tensor after zero-padding (shape=[batch_size, max_seq_len]={x.shape}):")
    print(x)

# Step 3: Generate Pad Mask (mark pad_id positions)
    pad_mask = MaskGenerator.gen_pad_mask(x, pad_id)
    print(f"\n3. Pad Mask (True=pad_id={pad_id}, need masking; shape=[batch_size, max_seq_len]={pad_mask.shape}):")
    print(pad_mask)

# Step 4: Generate Tgt Mask (mask future tokens, upper triangle)
    tgt_mask = MaskGenerator.gen_tgt_mask(x)
    print(f"\n4. Tgt Mask (True=future tokens, need masking; shape=[max_seq_len, max_seq_len]={tgt_mask.shape}):")
    print(tgt_mask)

# Validation Logic (ensure correct functionality)

    # Validate Pad Mask: All padded positions are True
    for i, seq in enumerate(raw_seqs):
        assert not pad_mask[i, :len(seq)].any(), f"Sample {i+1}: Real data was incorrectly masked"
        assert pad_mask[i, len(seq):].all(), f"Sample {i+1}: Padded zeros were not masked"

    # Validate Tgt Mask: Upper triangle logic
    seq_len = x.size(1)
    for i in range(seq_len):
        assert not tgt_mask[i, :i+1].any(), f"Token {i+1}: Current/historical tokens were incorrectly masked"
        assert tgt_mask[i, i+1:].all(), f"Token {i+1}: Future tokens were not masked"

    print("\n🎉 All test flows passed! Dual mask generation meets expectations.")


if __name__ == "__main__":
    main()


