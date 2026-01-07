import os, pathlib, toml
from .config import config


def create_variable(config):
    _path_data_dir = config["dir"]["data_dir"]
    if not _path_data_dir:
        proj_root = pathlib.Path(__file__).parent.parent
        _path_data_dir = os.path.join(proj_root, "data")

    class Variables():
        # def __init__(self):
            # print(f"data_dir : {self.path_data_dir}")
            # print("-" * 40)
            # print(
            #     f"epoch_total={self.epoch_total:02d}",
            #     "|",
            #     f"epoch_train={self.epoch_train:02d}",
            #     "|",
            #     f"epoch_eval={self.epoch_eval:02d}",
            #     )
            # print(
            #     f"batch_size={self.batch_size}",
            #     "|",
            #     f"seq_len={self.seq_len}",
            #     "|",
            #     f"d_model={self.d_model}",
            #     )
            # print(
            #     f"n_heads={self.n_heads}",
            #     "|",
            #     f"enc_n_layers={self.enc_n_layers}",
            #     "|",
            #     f"dec_n_layers={self.dec_n_layers}",
            #     )
            # print("-" * 40)
        path_data_dir = _path_data_dir

    # {{{ transformer
        d_model      = config["transformer"]["d_model"]
        d_ff         = config["transformer"]["d_ff"]
        n_heads      = config["transformer"]["n_heads"]
        enc_n_layers = config["transformer"]["enc_n_layers"]
        dec_n_layers = config["transformer"]["dec_n_layers"]
        src_dropout  = config["transformer"]["src_dropout"]
        tgt_dropout  = config["transformer"]["tgt_dropout"]
        enc_need_weight = config["transformer"]["enc_need_weight"]
        dec_need_weight = config["transformer"]["dec_need_weight"]
        src_vocab_size = config["transformer"]["src_vocab_size"]
        tgt_vocab_size = config["transformer"]["tgt_vocab_size"]

    # {{{ path
        path_tokenid_src_train = os.path.join(path_data_dir, config["file"]["tokenid_src_train"])
        path_tokenid_tgt_train = os.path.join(path_data_dir, config["file"]["tokenid_tgt_train"])
        path_tokenid_src_eval  = os.path.join(path_data_dir, config["file"]["tokenid_src_eval"] )
        path_tokenid_tgt_eval  = os.path.join(path_data_dir, config["file"]["tokenid_tgt_eval"] )
        path_vocab_src         = os.path.join(path_data_dir, config["file"]["vocab_src"]        )
        path_vocab_tgt         = os.path.join(path_data_dir, config["file"]["vocab_tgt"]        )
        path_tokenizer_src     = os.path.join(path_data_dir, config["file"]["tokenizer_src"]    )
        path_tokenizer_tgt     = os.path.join(path_data_dir, config["file"]["tokenizer_tgt"]    )
        path_model_log         = os.path.join(path_data_dir, config["file"]["model_log"]        )
        path_model_weight      = os.path.join(path_data_dir, config["file"]["model_weight"]     )
        path_model_weight_new  = os.path.join(path_data_dir, config["file"]["model_weight_new"] )

    # {{{ general
        device    = config["general"]["device"]
        text      = config["general"]["text"]
        input_text  = config["general"]["input_text"]
        output_text = config["general"]["output_text"]

    # {{{ epoch
        current_epoch = config["epoch"]["current_epoch"]
        total_stage_epoch = config["epoch"]["total_stage_epoch"]
        total_train_epoch = config["epoch"]["total_train_epoch"]
        total_eval_epoch  = config["epoch"]["total_eval_epoch"]
        total_infer_epoch = config["epoch"]["total_infer_epoch"]

    # {{{ vocab
        unk_token = config["vocab"]["unk_token"]
        pad_token = config["vocab"]["pad_token"]
        sos_token = config["vocab"]["sos_token"]
        eos_token = config["vocab"]["eos_token"]
        unk_id    = config["vocab"]["unk_id"]
        pad_id    = config["vocab"]["pad_id"]
        sos_id    = config["vocab"]["sos_id"]
        eos_id    = config["vocab"]["eos_id"]

    # {{{ data
        batch_size    = config["dataset"]["batch_size"]
        max_seq_len   = config["dataset"]["max_seq_len"]
        shuffle_train = config["dataset"]["shuffle_train"]
        shuffle_eval  = config["dataset"]["shuffle_eval"]
        tokenizer_src = config["dataset"]["tokenizer_src"]
        tokenizer_tgt = config["dataset"]["tokenizer_tgt"]
        id2word_src      = config["dataset"]["id2word_src"]
        id2word_tgt      = config["dataset"]["id2word_tgt"]
        databatchs_train = config["dataset"]["databatchs_train"]
        databatchs_eval  = config["dataset"]["databatchs_eval"]

    # {{{ para
        optim_lr            = float(config["optimizer"]["lr"]          )
        optim_weight_decay  = float(config["optimizer"]["weight_decay"])
        sched_min_lr        = float(config["scheduler"]["min_lr"]      )
        optim_betas         = config["optimizer"]["betas"]
        sched_mode          = config["scheduler"]["mode"]
        sched_factor        = config["scheduler"]["factor"]
        sched_patience      = config["scheduler"]["patience"]
        sched_verbose       = config["scheduler"]["verbose"]
        loss_label_smoothing= config["metricmeter"]["label_smoothing"]
        bleu_weights        = config["metricmeter"]["bleu_weights"]
        penalty_ids         = config["penalizer"]["penalty_ids"]
        penalty_weight      = config["penalizer"]["penalty_weight"]
        stop_patience_loss  = config["stopper"]["patience_loss"]
        stop_patience_bleu  = config["stopper"]["patience_bleu"]
        stop_delta_loss     = config["stopper"]["delta_loss"]
        stop_delta_bleu     = config["stopper"]["delta_bleu"]

    # {{{ regulator
        optimizer    = None
        scheduler    = None
        metricmeter  = None
        penalizer    = None
        loss_stopper = None
        bleu_stopper = None

    ##
    return Variables


Variables = create_variable(config)


