import os, torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from tokenizers import Tokenizer
from tqdm import tqdm
from utils import load_config, create_variable, init_logger
from transformer.model import Transformer
from transformer.utils import (
    DataHandler, MetricMeter, LossPenalizer, EarlyStopper, ModelHandler, )


print("=" * 40)
print(f"-- Config to Variable --")
print("=" * 40)
Variables = create_variable(load_config())

def init() -> tuple[Variables, Transformer, DataHandler, ModelHandler]:
    print("=" * 40)
    print(f"-- Transformer Model Initialization --")
    print("=" * 40)
    v = Variables()

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

def train(v:Variables, model:Transformer, dh:DataHandler, mh:ModelHandler):
    e_total = v.current_epoch_total
    for e_train in range(1, v.epoch_train+1):
        print("=" * 40)
        generator = mh.train_model(v.databatchs_train, v.optimizer, v.metricmeter, v.penalizer)
        for data in generator :
            if data["type"] == "init":
                batch = data["data"]["batch"]
                desc  = data["data"]["desc"]
                pbar = tqdm(total=batch, desc=desc, unit="step", ncols=0, dynamic_ncols=True)
            elif data["type"] == "update":
                loss    = data["data"]["loss"]
                ids_pen = data["data"]["ids_pen"]
                pbar.postfix = f"loss={loss:.4f}, ids_pen={ids_pen:.4f}"
                pbar.update(1)
            elif data["type"] == "final":
                pbar.close()
                loss    = data["data"]["loss"]
                ids_pen = data["data"]["ids_pen"]
                print(f"| loss={loss:.4f} | ids_pen={ids_pen:.4f}")

        print("-" * 40)
        log_content = \
            f" | Total: {e_total:02d}/{v.epoch_total:02d}" \
            f" | Train: {e_train:02d}/{v.epoch_train:02d}" \
            f" | loss: {loss:.4f}"
        dh.save_log(log_content, v.path_model_log)
        dh.save_model_weight(model, v.path_model_weight_new)

def eval(v:Variables, model:Transformer, dh:DataHandler, mh:ModelHandler):
    e_total = v.current_epoch_total
    for e_eval  in range(1, v.epoch_eval+1):
        print("=" * 40)
        generator = mh.eval_model(v.databatchs_eval, v.scheduler, v.metricmeter)
        for data in generator :
            if data["type"] == "init":
                batch = data["data"]["batch"]
                desc  = data["data"]["desc"]
                pbar = tqdm(total=batch, desc=desc, unit="step", ncols=0, dynamic_ncols=True)
            elif data["type"] == "update":
                loss = data["data"]["loss"]
                bleu = data["data"]["bleu"]
                pbar.postfix = f"loss={loss:.4f}, bleu={bleu:.4f}"
                pbar.update(1)
            elif data["type"] == "final":
                pbar.close()
                loss = data["data"]["loss"]
                bleu = data["data"]["bleu"]
                print(f"| loss={loss:.4f} | bleu={bleu:.4f}")

        print("-" * 40)
        log_content = \
            f" | Total: {e_total:02d}/{v.epoch_total:02d}" \
            f" | Eval:  {e_eval:02d}/{v.epoch_eval:02d}" \
            f" | loss: {loss:.4f}" f" | bleu: {bleu:.4f}"
        dh.save_log(log_content, v.path_model_log)
        v.loss_stopper.track_metric(loss)
        v.bleu_stopper.track_metric(bleu)
        dh.save_model_weight(model, v.path_model_weight)

def infer(v:Variables, mh:ModelHandler):
    for e_infer in range(1, v.epoch_infer+1):
        print("=" * 40)
        print(f"Model Infer:")

        print("-" * 40)
        print(v.text)
        text_gen = mh.infer_model(v.text, v.seq_len,
            v.pad_id, v.unk_id, v.sos_id, v.eos_id,
            v.tokenizer_src,
            v.tokenizer_tgt,
            )
        for word in text_gen :
            print(word, end="", flush=True)
        print()


def main():
    v, m, dh, mh = init()

    log_content = f" {('-' * 50)}"
    dh.save_log(log_content, v.path_model_log)

    for e_total in range(1, v.epoch_total+1):
        print("#" * 50)
        if v.epoch_total > 1:
            v.current_epoch_total = e_total

        train(v, m, dh, mh)
        eval(v, m, dh, mh)
        infer(v, mh)


if __name__ == "__main__":
    main()


