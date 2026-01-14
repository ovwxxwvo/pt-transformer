import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from common import load_config
from transformer import create_variable


def main():
    print(f"-- Test Variable Module --")
    print("=" * 40)

    print()
    v = create_variable(load_config())

    print(f"path_data_dir={v.path_data_dir}")
    print(f"total_stage_epoch={v.total_stage_epoch}, total_train_epoch={v.total_train_epoch}, total_eval_epoch={v.total_eval_epoch}")
    print(f"{('-' * 50)}")
    print(f"device={v.device}")
    print(f"unk_id={v.unk_id}, pad_id={v.pad_id}, sos_id={v.sos_id}, eos_id={v.eos_id}")
    print(f"{('-' * 50)}")
    print(f"batch_size={v.batch_size}, max_seq_len={v.max_seq_len}, d_model={v.d_model}")
    print(f"n_heads={v.n_heads}, enc_n_layers={v.enc_n_layers}, dec_n_layers={v.dec_n_layers}")
    print(f"{('-' * 50)}")
    print(f"optem: lr={v.optim_lr}, weight_decay={v.optim_weight_decay}")
    print(f"sched: factor={v.sched_factor}, patience={v.sched_patience}, min_lr={v.sched_min_lr}")
    print(f"{('-' * 50)}")

    print("\n🎉 Variable Module test passed completely!")


if __name__ == "__main__":
    main()


