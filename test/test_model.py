import torch
from transformer.model import Transformer


def main():
    print(f"-- Test Model Workflow --")
    print("=" * 40)

# Step 1: Data Initialization (Generate test tensors + assign key parameters)
    print("\n【Step 1: Data Initialization】")
    src_vocab_size = 1024
    tgt_vocab_size = 1024
    batch_size  = 8
    src_seq_len = 64
    tgt_seq_len = 64
    # Generate tensors (values avoid pad_id, range: [1, vocab_size-1])
    x_src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len), dtype=torch.long)
    x_tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len), dtype=torch.long)
    # Key parameters (loaded from config, no manual definition needed)
    print(f"✓ Key params: src_vocab_size={src_vocab_size} | tgt_vocab_size={tgt_vocab_size}")
    print(f"✓ Tensors generated: x_src({x_src.shape}) | x_tgt({x_tgt.shape})")

# Step 2: Model Initialization (keep your original logic, fix param order)
    print("\n【Step 2: Model Initialization】")
    pad_id = 0
    n_heads = 4
    enc_n_layers = 8
    dec_n_layers = 8
    enc_need_weight = True
    dec_need_weight = True
    model: torch.nn.Module = Transformer(
        pad_id,
        src_vocab_size,
        tgt_vocab_size,
        n_heads=n_heads,
        enc_n_layers=enc_n_layers,
        dec_n_layers=dec_n_layers,
        enc_need_weight=enc_need_weight,
        dec_need_weight=dec_need_weight,
        )
    print(f"✓ Model initialized successfully")

# Step 3: Model Testing (Forward pass + result validation)
    print("\n【Step 3: Model Testing】")
    model.eval()           # Switch to eval mode (disable Dropout)
    with torch.no_grad():  # Disable gradient computation for efficiency
        logits, attn_weights = model(x_src, x_tgt)

# Result validation (core metrics)

    # 1. Validate output shape
    expected_shape = (batch_size, tgt_seq_len, tgt_vocab_size)
    assert logits.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {logits.shape}"
    # 2. Validate no abnormal values (NaN/Inf)
    assert not torch.isnan(logits).any(), "Output contains NaN! Model is unstable"
    assert not torch.isinf(logits).any(), "Output contains Inf! Model is unstable"
    # 3. 核心权重验证
    if enc_need_weight:
        assert len(attn_weights["enc"]) == enc_n_layers, f"Encoder weight layers mismatch! Expected {enc_n_layers}, got {len(attn_weights['enc'])}"
        assert all(isinstance(layer_w[0], torch.Tensor) for layer_w in attn_weights["enc"]), "Encoder has invalid weight (not tensor)"
    if dec_need_weight:
        assert len(attn_weights["dec"]) == dec_n_layers, f"Decoder weight layers mismatch! Expected {dec_n_layers}, got {len(attn_weights['dec'])}"
        assert all(isinstance(layer_w[0], torch.Tensor) for layer_w in attn_weights["dec"]), "Decoder has invalid weight (not tensor)"

    print(f"✓ Forward pass successful! Logits shape: {logits.shape}")
    print(f"✓ Value validation passed (No NaN/Inf)")
    if enc_need_weight or dec_need_weight:
        print(f"✓ Attention weights output normally")
        print(f"✓ Encoder attention: {len(attn_weights['enc'])} layers (valid tensors)")
        print(f"✓ Decoder attention: {len(attn_weights['dec'])} layers (valid tensors)")
    print("\n🎉 Transformer model test passed completely!")


if __name__ == "__main__":
    main()


