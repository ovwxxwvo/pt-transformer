import time
import torch
from model import Transformer


def main():
    d_model = 512
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    enc_n_layers = 8
    dec_n_layers = 8
    batch_size  = 8

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    model = Transformer(
        d_model=d_model,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        enc_n_layers=enc_n_layers,
        dec_n_layers=dec_n_layers
    # )
    ).to(device)

    src_seq_len = 256
    tgt_seq_len = 256
    # x_src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    # x_tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    x_src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len)).to(device)
    x_tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)

    # src_mask = MaskGenerator.gen_pad_mask(x_src)
    # tgt_mask = MaskGenerator.gen_tgt_mask(x_tgt)

    model.eval()
    with torch.no_grad():
        # output = model(x_src, x_tgt, src_mask, tgt_mask)
        rounds = 20
        for round in range(rounds):
            torch.cuda.empty_cache()
            output = model(x_src, x_tgt)
            time.sleep(1)
            if (round + 1) % 5 == 0:
                print(f"完成 {round + 1:02d}/{rounds} 轮 | 输出形状: {output.shape}")

    expected_shape = (batch_size, tgt_seq_len, tgt_vocab_size)
    if output.shape == expected_shape:
        print("Test passed! Output shape:", output.shape)
    else:
        print("Test failed!")
        print("Actual shape:", output.shape)
        print("Expected shape:", expected_shape)


if __name__ == "__main__":
    main()


