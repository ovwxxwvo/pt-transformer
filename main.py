import os, argparse, torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from tokenizers import Tokenizer
from tqdm import tqdm
from transformer import ( Transformer,
    DataHandler, MetricMeter, LossPenalizer, EarlyStopper, ModelHandler, )
from utils import ( Variables,
    model_logger, main_logger, server_logger,
    get_metric_db, parse_cli_args,
    )


def init(args:argparse.Namespace) -> tuple[Variables, Transformer, DataHandler, ModelHandler]:
    # init variable
    v = Variables()
    if args.total_stage_epoch is not None: v.total_stage_epoch = args.total_stage_epoch
    if args.total_train_epoch is not None: v.total_train_epoch = args.total_train_epoch
    if args.total_eval_epoch  is not None: v.total_eval_epoch  = args.total_eval_epoch
    if args.total_infer_epoch is not None: v.total_infer_epoch = args.total_infer_epoch

    main_logger.info("Vars Initializing ...")
    main_logger.info(f"path_data_dir={v.path_data_dir}")
    main_logger.info(
        f"total_stage_epoch={v.total_stage_epoch}, total_train_epoch={v.total_train_epoch}, total_eval_epoch={v.total_eval_epoch}" \
        )
    main_logger.info(f"device={v.device}")
    main_logger.info(f"{('-' * 50)}")

    # init data
    dh = DataHandler(v.path_data_dir)

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

    main_logger.info("Data Initializing ...")
    main_logger.info(f"unk_id={v.unk_id}, pad_id={v.pad_id}, sos_id={v.sos_id}, eos_id={v.eos_id}")
    main_logger.info(f"src_vocab_size={v.src_vocab_size}, tgt_vocab_size={v.tgt_vocab_size}")
    main_logger.info(f"{('-' * 50)}")

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

    main_logger.info("Model Initializing ...")
    main_logger.info(f"batch_size={v.batch_size}, max_seq_len={v.max_seq_len}, d_model={v.d_model}")
    main_logger.info(f"n_heads={v.n_heads}, enc_n_layers={v.enc_n_layers}, dec_n_layers={v.dec_n_layers}")
    if os.path.exists(v.path_model_weight):
        main_logger.info(f"Loaded pre-trained model from {v.path_model_weight}")
    else:
        main_logger.info("No pre-trained model found, initializing model from scratch")
    main_logger.info(f"{('-' * 50)}")

    # init regulator
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

    main_logger.info("Regulator Initializing ...")
    main_logger.info(f"optem: lr={v.optim_lr}, weight_decay={v.optim_weight_decay}")
    main_logger.info(f"sched: factor={v.sched_factor}, patience={v.sched_patience}, min_lr={v.sched_min_lr}")
    main_logger.info(f"{('-' * 50)}")

    # return
    return v, model, dh, mh

def train(v:Variables, model:Transformer, dh:DataHandler, mh:ModelHandler):
    stage_epoch = v.current_epoch
    total_stage_epoch = v.total_stage_epoch
    total_task_epoch  = v.total_train_epoch

    for epoch in range(1, total_task_epoch+1):
        print("-" * 40)
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
                # print(f"| loss={loss:.4f} | ids_pen={ids_pen:.4f}")

        loss = round(loss, 4)
        msg = \
            f"Stage: {stage_epoch:02d}/{total_stage_epoch:02d} | " \
            f"Train: {epoch:02d}/{total_task_epoch:02d} | " \
            f"loss={loss:.4f}, ids_pen={ids_pen:.4f}"
        model_logger.info(msg)

        db = get_metric_db()
        db.insert_metric(step_type="train",
            stage_epoch=stage_epoch, total_stage_epoch=total_stage_epoch,
            task_epoch=epoch, total_task_epoch=total_task_epoch,
            loss=loss, bleu=None )

        dh.save_model_weight(model, v.path_model_weight_new)

def eval(v:Variables, model:Transformer, dh:DataHandler, mh:ModelHandler):
    stage_epoch = v.current_epoch
    total_stage_epoch = v.total_stage_epoch
    total_task_epoch  = v.total_eval_epoch

    for epoch in range(1, total_task_epoch+1):
        print("-" * 40)
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
                # print(f"| loss={loss:.4f} | bleu={bleu:.4f}")

        loss = round(loss, 4)
        bleu = round(bleu, 4)
        msg = \
            f"Stage: {stage_epoch:02d}/{total_stage_epoch:02d} | " \
            f"Eval:  {epoch:02d}/{total_task_epoch:02d} | " \
            f"loss={loss:.4f}, bleu={bleu:.4f}"
        model_logger.info(msg)

        db = get_metric_db()
        db.insert_metric(step_type="eval",
            stage_epoch=stage_epoch, total_stage_epoch=total_stage_epoch,
            task_epoch=epoch, total_task_epoch=total_task_epoch,
            loss=loss, bleu=bleu )

        v.loss_stopper.track_metric(loss)
        v.bleu_stopper.track_metric(bleu)
        dh.save_model_weight(model, v.path_model_weight)

def infer(v:Variables, mh:ModelHandler):
    # stage_epoch = v.current_epoch
    # total_stage_epoch = v.total_stage_epoch
    total_task_epoch  = v.total_infer_epoch

    for epoch in range(1, total_task_epoch+1):
        print("-" * 40)
        # print(f"Model Infer:")

        text_gen = mh.infer_model(v.text, v.max_seq_len,
            v.pad_id, v.unk_id, v.sos_id, v.eos_id,
            v.tokenizer_src,
            v.tokenizer_tgt,
            )
        # print(v.text)
        # for word in text_gen :
        #     print(word, end="", flush=True)
        # print()

        text_gen = "".join(list(text_gen)) if text_gen else ""
        print(f"Model Infer: {v.text} | {text_gen}")

        msg = f"{v.text} | {text_gen}"
        main_logger.info(msg)

def pipeline(v:Variables, m:Transformer, dh:DataHandler, mh:ModelHandler):
    model_logger.info(f"{('-' * 50)}")
    total_stage_epoch = v.total_stage_epoch

    for epoch in range(1, total_stage_epoch+1):
        print("=" * 80)
        v.current_epoch = epoch
        msg = \
            f"Total: {epoch:02d}/{total_stage_epoch:02d} | " \
            f"train & eval & infer"
        main_logger.info(msg)

        train(v, m, dh, mh)
        eval(v, m, dh, mh)
        infer(v, mh)

def main():
    args = parse_cli_args()

    main_logger.info(f"{('-' * 50)}")
    v, m, dh, mh = init(args)

    match args.mode:
        case "all":
            pipeline(v, m, dh, mh)
        case "train":
            v.current_epoch = 0; v.total_stage_epoch = 0
            train(v, m, dh, mh)
        case "eval":
            v.current_epoch = 0; v.total_stage_epoch = 0
            eval(v, m, dh, mh)
        case "infer":
            v.current_epoch = 0; v.total_stage_epoch = 0
            infer(v, mh)
        case _:
            raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()


