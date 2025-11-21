import os, torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from tokenizers import Tokenizer
from tqdm import tqdm
from utils import config, variable
from transformer.model import Transformer
from transformer.utils import (
    DataHandler, MetricMeter, LossPenalizer, EarlyStopper, ModelHandler, )


def init():
    print(f"-- Transformer Model --")
    print("=" * 40)
    Variable = variable.assign(config.load())
    v = Variable()

    # init data
    dh = DataHandler(v.path_data_dirt)

    v.tokenizer_src = Tokenizer.from_file(v.path_tokenizer_src)
    v.tokenizer_tgt = Tokenizer.from_file(v.path_tokenizer_tgt)

    v.pad_id, v.unk_id, v.sos_id, v.eos_id = dh.get_special_id(
        v.tokenizer_src,
        v.unk_token, v.pad_token, v.sos_token, v.eos_token
        )
    v.src_vocab_size = dh.get_vocab_size(v.tokenizer_src)
    v.tgt_vocab_size = dh.get_vocab_size(v.tokenizer_tgt)
    v.id2word_tgt = dh.reverse_vocab(v.tokenizer_tgt)

    v.databatchs_train = dh.batch_data(
        v.path_tokenid_src_train,
        v.path_tokenid_tgt_train,
        v.batch_size, v.shuffle_train,
        )
    v.databatchs_eval  = dh.batch_data(
        v.path_tokenid_src_eval,
        v.path_tokenid_tgt_eval,
        v.batch_size, v.shuffle_eval,
        )

    # init model
    model:torch.nn.Module = Transformer(
        v.pad_id,
        v.src_vocab_size,
        v.tgt_vocab_size,
        d_model=v.d_model,
        n_heads=v.n_heads,
        enc_n_layers=v.enc_n_layers,
        dec_n_layers=v.dec_n_layers,
        d_ff=v.d_ff,
        src_dropout=v.src_dropout,
        tgt_dropout=v.tgt_dropout,
        enc_need_weight=v.enc_need_weight,
        dec_need_weight=v.dec_need_weight,
        ).to(v.device)
    if os.path.exists(v.path_model_weight):
        model.load_state_dict(torch.load(
            v.path_model_weight,
            map_location=v.device,
            ))

    mh = ModelHandler(model, device=v.device)

    # model regulator
    v.optimizer = optim.AdamW(
        model.parameters(),
        lr=v.optim_lr,
        weight_decay=v.optim_weight_decay,
        betas=v.optim_betas,
        )
    v.scheduler = sched.ReduceLROnPlateau(
        v.optimizer,
        mode=v.sched_mode,
        patience=v.sched_patience,
        factor=v.sched_factor,
        min_lr=v.sched_min_lr,
        # verbose=v.verbose,
        )
    v.metricmeter = MetricMeter(
        v.pad_id, v.unk_id, v.sos_id, v.eos_id,
        label_smoothing=v.loss_label_smoothing,
        id2word_tgt=v.id2word_tgt,
        bleu_weights=v.bleu_weights,
        )
    v.penalizer = LossPenalizer(
        v.pad_id,
        penalty_ids=v.penalty_ids,
        penalty_weight=v.penalty_weight,
        # entropy_enc_weight=v.entropy_enc_weight,
        # entropy_dec_weight=v.entropy_dec_weight,
        )
    v.loss_stopper = EarlyStopper("loss",
        patience=v.stop_patience_loss,
        delta=v.stop_delta_loss,
        )
    v.bleu_stopper = EarlyStopper("bleu",
        patience=v.stop_patience_bleu,
        delta=v.stop_delta_bleu,
        )

    # return
    return v, model, dh, mh


if __name__ == "__main__":
    main()


