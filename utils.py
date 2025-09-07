import os, json, toml, torch
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tokenizers import Tokenizer


class DataHandler():
    def __init__(self, path_data_dir):
        self.data_dir = path_data_dir

    def get_vocab_size(self, path_tokenid):
        x = torch.load(path_tokenid)
        vocab_size = x.max().item() + 1
        print(
            f"x shape : {tuple(x.shape)}",
            f"|",
            f"vocab size : {vocab_size}",
            )
        # print("-" * 40)
        return vocab_size

    def reverse_vocab(self, path_vocab):
        with open(path_vocab, "r", encoding="utf-8") as f:
            old_dict = json.load(f)
        new_dict = {str(v): str(k) for k, v in old_dict.items()}
        print(f"Vocab reversed.")
        print("=" * 40)
        return new_dict

    def get_pad_id(self, path_tokenizer, token="[PAD]"):
        tokenizer = Tokenizer.from_file(path_tokenizer)
        pad_id = tokenizer.token_to_id(token)
        # print("tokenizer", tokenizer)
        # print("pad_id :", pad_id)
        return pad_id

    def batch_data(self,
            path_tokenid_src,
            path_tokenid_tgt,
            batch_size, data_shuffle,
            ):
        x_src = torch.load(path_tokenid_src)
        x_tgt = torch.load(path_tokenid_tgt)
        print(f"src x shape : {tuple(x_src.shape)}")
        print(f"tgt x shape : {tuple(x_tgt.shape)}")
        print("-" * 40)
        # truncated_len = 128
        # x_src = x_src[:, :truncated_len]
        # x_tgt = x_tgt[:, :truncated_len]
        dataset = TensorDataset(x_src, x_tgt)
        databatchs = DataLoader(dataset,
            batch_size=batch_size, shuffle=data_shuffle, drop_last=True)
        print(f"Data batched.")
        print("=" * 40)
        return databatchs

    def save_log(self, log_content, save_name="model.log"):
        data_dir = self.data_dir
        save_path = os.path.join(data_dir, save_name)
        current_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        log_line = f"[{current_time}] | {log_content}\n"
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(log_line)
        # print(f"{log_line.strip()}")
        # print(f"Model Log saved, {save_path}")

    def save_model_weight(self, model, save_name="model_weight.pth"):
        data_dir = self.data_dir
        save_path = os.path.join(data_dir, save_name)
        torch.save(model.state_dict(), save_path)
        print("=" * 40)
        print(f"Model Weights saved, {save_path}")

class LossMeter():
    def __init__(self, sample_num, pad_id):
        self.total_loss = 0.0
        self.avg_loss = 0.0
        self.sample_num = sample_num
        self.seq_loss = torch.nn.NLLLoss(ignore_index=pad_id)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def cal_loss(self, logits, x_tgt):
        sample_num = self.sample_num
        log_probs = self.log_softmax(logits)
        flat_log_probs = log_probs.reshape(-1, log_probs.size(-1))
        flat_true_tgt = x_tgt.reshape(-1)
        loss = self.seq_loss(flat_log_probs, flat_true_tgt)
        self.total_loss += loss.item() * x_tgt.size(0)
        self.avg_loss = self.total_loss / self.sample_num
        return loss

class BleuMeter():
    def __init__(self, sample_num, pad_id, id2word):
        self.total_bleu = 0.0
        self.avg_bleu = 0.0
        self.sample_num = sample_num
        self.pad_id = pad_id
        self.id2word = id2word
        self.smooth_fn = SmoothingFunction().method4

    def cal_bleu(self, logits, x_tgt):
        sample_num = self.sample_num
        pad_id     = self.pad_id
        id2word    = self.id2word
        smooth_fn = self.smooth_fn
        batch_bleu = []
        x_pred = logits.argmax(dim=-1)
        for pred_ids, tgt_ids in zip(x_pred, x_tgt):
            valid_tgt_ids  = [id.item() for id in tgt_ids if id.item() != pad_id]
            valid_pred_ids = [id.item() for id in pred_ids if id.item() != pad_id]
            true_text = "".join([id2word.get(str(id), "<unk>") for id in valid_tgt_ids])
            pred_text = "".join([id2word.get(str(id), "<unk>") for id in valid_pred_ids])
            references = [list(true_text)]
            candidate  = list(pred_text)
            bleu = sentence_bleu(references, candidate, smoothing_function=smooth_fn)
            batch_bleu.append(bleu)
        bleu = sum(batch_bleu) / len(batch_bleu)
        self.total_bleu += bleu * x_tgt.size(0)
        self.avg_bleu = self.total_bleu / self.sample_num
        return bleu

class MetricMeter(LossMeter, BleuMeter):
    def __init__(self, sample_num, pad_id, id2word={}):
        LossMeter.__init__(self, sample_num, pad_id)
        BleuMeter.__init__(self, sample_num, pad_id, id2word)

class ModelHandler():
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def train_model(self, databatchs, optimizer, metricmeter):
        device = self.device
        model = self.model
        model.train()
        metric = metricmeter
        print(f"Model Train ...")
        print("=" * 40)

        batch_count = len(databatchs)
        # print(f"batch: {batch:02d}")
        pbar = tqdm(total=batch_count, desc="batch", unit=" step", ncols=0, dynamic_ncols=True)

        # sample_num = len(databatchs.dataset)
        # seq_loss = SequenceLoss(sample_num, pad_id)
        for x_src, x_tgt in databatchs:
            x_src =  x_src.to(device)
            x_tgt =  x_tgt.to(device)
            optimizer.zero_grad()
            logits = model(x_src, x_tgt)
            loss = metric.cal_loss(logits, x_tgt)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss:.4f}")
            pbar.update(1)
            # time.sleep(0.2)

        pbar.close()
        avg_loss = metric.avg_loss
        print(f"loss:{avg_loss:.4f}")
        print("-" * 40)

        print("=" * 40)
        return avg_loss

    def eval_model( self, databatchs, scheduler, metricmeter):
        device = self.device
        model = self.model
        model.eval()
        metric = metricmeter
        print(f"Model Eval ...")
        print("=" * 40)

        batch = len(databatchs)
        # print(f"batch: {batch:02d}")
        pbar = tqdm(total=batch, desc="batch", unit=" step", ncols=0, dynamic_ncols=True)

        torch.no_grad()
        for x_src, x_tgt in databatchs:
            x_src =  x_src.to(device)
            x_tgt =  x_tgt.to(device)
            logits = model(x_src, x_tgt)
            loss = metric.cal_loss(logits, x_tgt)
            bleu = metric.cal_bleu(logits, x_tgt)
            pbar.set_postfix(loss=f"{loss:.4f}", bleu=f"{bleu:.4f}")
            pbar.update(1)
            # time.sleep(0.2)

        pbar.close()
        avg_loss = metric.avg_loss
        avg_bleu = metric.avg_bleu
        scheduler.step(avg_loss)
        print(f"loss:{avg_loss:.4f}", f"bleu:{avg_bleu:.4f}")
        print("-" * 40)

        print("=" * 40)
        return avg_loss, avg_bleu



# class MetricMeter():
    # def __init__(self, sample_num, pad_id, id2word=None):
    #     self.total_loss = 0.0
    #     self.total_bleu = 0.0
    #     self.avg_loss = 0.0
    #     self.avg_bleu = 0.0
    #     self.sample_num = sample_num
    #     self.pad_id     = pad_id
    #     self.id2word    = id2word
    #     self.seq_loss    = torch.nn.NLLLoss(ignore_index=pad_id)
    #     self.log_softmax = torch.nn.LogSoftmax(dim=-1)
    #
    # def cal_loss(self, logits, x_tgt):
    #     sample_num = self.sample_num
    #     log_probs = self.log_softmax(logits)
    #     flat_log_probs = log_probs.reshape(-1, log_probs.size(-1))
    #     flat_true_tgt = x_tgt.reshape(-1)
    #     loss = self.seq_loss(flat_log_probs, flat_true_tgt)
    #     self.total_loss += loss.item() * x_tgt.size(0)
    #     self.avg_loss = self.total_loss / self.sample_num
    #     return loss
    #
    # def cal_bleu(self, logits, x_tgt):
    #     sample_num = self.sample_num
    #     pad_id     = self.pad_id
    #     id2word    = self.id2word
    #     batch_bleu = []
    #     x_pred = logits.argmax(dim=-1)
    #     smooth_fn = SmoothingFunction().method4
    #     for pred_ids, tgt_ids in zip(x_pred, x_tgt):
    #         valid_tgt_ids  = [id.item() for id in tgt_ids if id.item() != pad_id]
    #         valid_pred_ids = [id.item() for id in pred_ids if id.item() != pad_id]
    #         true_text = "".join([id2word.get(str(id), "<unk>") for id in valid_tgt_ids])
    #         pred_text = "".join([id2word.get(str(id), "<unk>") for id in valid_pred_ids])
    #         references = [list(true_text)]
    #         candidate  = list(pred_text)
    #         bleu = sentence_bleu(references, candidate, smoothing_function=smooth_fn)
    #         batch_bleu.append(bleu)
    #     bleu = sum(batch_bleu) / len(batch_bleu)
    #     self.total_bleu += bleu * x_tgt.size(0)
    #     self.avg_bleu = self.total_bleu / self.sample_num
    #     return bleu

# class ConfigLoader():
    # # def __init__(self,):
    #
    # def load_config():
    #     dirt_path = os.path.dirname(os.path.abspath(__file__))
    #     print(dirt_path)
    #     with open("config.toml", "r") as f:
    #         config = toml.load(f)
    #     # print(config)
    #     config_names = config["include"]["config_names"]
    #     # print(config_names)
    #     for name in config_names:
    #         path = os.path.join(dirt_path, "config", name)
    #         # print(path)
    #         with open(path, "r") as f:
    #             config.update(toml.load(f))
    #     # print(config)
    #     print("=" * 40)
    #     return config
    #
    # def set_var():
    #     path_tokenid_src_train = config["path"]["path_tokenid_src_train"]
    #     path_tokenid_tgt_train = config["path"]["path_tokenid_tgt_train"]
    #     path_tokenid_src_eval  = config["path"]["path_tokenid_src_eval"]
    #     path_tokenid_tgt_eval  = config["path"]["path_tokenid_tgt_eval"]
    #     path_vocab_tgt_eval    = config["path"]["path_vocab_tgt_eval"]
    #     path_model_weight      = config["path"]["path_model_weight"]
    #     d_model      = config["transformer"]["d_model"]
    #     enc_n_heads  = config["transformer"]["enc_n_heads"]
    #     dec_n_heads  = config["transformer"]["dec_n_heads"]
    #     enc_n_layers = config["transformer"]["enc_n_layers"]
    #     dec_n_layers = config["transformer"]["dec_n_layers"]
    #     batch_size   = config["dataset"]["batch_size"]
    #     shuffle_eval = config["dataset"]["shuffle_eval"]
    #

class EarlyStopping:
    def __init__(self, patience=2, best_model_path="best_bleu_model.pth"):
        self.best_bleu = 0.0
        self.patience = patience
        self.counter = 0
        self.stop_flag = False
        self.save_path = best_model_path


