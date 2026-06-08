import os, sys, math, torch
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from .monitor import MetricMeter, LossPenalizer


class DataHandler():
    def __init__(self, path_data_dir):
        self.data_dir = path_data_dir

    def get_special_id(self, tokenizer,
            unk_token, pad_token, sos_token, eos_token, ):
        unk_id = tokenizer.token_to_id(unk_token)
        pad_id = tokenizer.token_to_id(pad_token)
        sos_id = tokenizer.token_to_id(sos_token)
        eos_id = tokenizer.token_to_id(eos_token)
        # print("tokenizer", tokenizer)
        # print(
            # f"pad_id={pad_id}", "|",
            # f"unk_id={unk_id}", "|",
            # f"sos_id={sos_id}", "|",
            # f"eos_id={eos_id}", "|",
            # )
        return pad_id, unk_id, sos_id, eos_id

    def get_vocab_size(self, tokenizer):
        vocab_size = tokenizer.get_vocab_size()
        # print(f"vocab size : {vocab_size}")
        # print("-" * 40)
        return vocab_size

    def reverse_vocab(self, tokenizer):
        vocab = tokenizer.get_vocab()
        new_vocab = {int(v): str(k) for k, v in vocab.items()}
        # print(f"Vocab reversed.")
        # print("-" * 40)
        return new_vocab

    def batch_data(self,
            path_tokenid_src,
            path_tokenid_tgt,
            batch_size, data_shuffle,
            ):
        x_src = torch.load(path_tokenid_src)
        x_tgt = torch.load(path_tokenid_tgt)
        # print(f"src x shape : {tuple(x_src.shape)}")
        # print(f"tgt x shape : {tuple(x_tgt.shape)}")
        # print("-" * 40)
        # truncated_len = 128
        # x_src = x_src[:, :truncated_len]
        # x_tgt = x_tgt[:, :truncated_len]
        dataset = TensorDataset(x_src, x_tgt)
        del x_src, x_tgt
        databatchs = DataLoader(dataset,
            batch_size=batch_size, shuffle=data_shuffle, drop_last=True)
        # print(f"Data batched.")
        # print("-" * 40)
        return databatchs

    def save_log(self, log_content, save_path=""):
        if not save_path:
            data_dir = self.data_dir
            save_name="model.log"
            save_path = os.path.join(data_dir, save_name)
        current_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        log_line = f"[{current_time}]{log_content}\n"
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(log_line)
        # print(f"{log_line.strip()}")
        # print(f"Model Log saved, {save_path}")

    def save_model_weight(self, model, save_path=""):
        if not save_path:
            data_dir = self.data_dir
            save_name="model_weight_new.pth"
            save_path = os.path.join(data_dir, save_name)
        torch.save(model.state_dict(), save_path)
        # print(f"Model Weights saved, {save_path}")
        # print("=" * 40)

    def check_tokenid(self, path_tokenid, id2word):
        x = torch.load(path_tokenid)
        ids = x[100].tolist()
        text_gen = [id2word[id] for id in ids]
        text_gen = " ".join(text_gen)
        # print(ids)
        # print(text_gen)

class ModelHandler():
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    @torch.enable_grad()
    def train_model(self, databatchs,
        optimizer:      torch.optim,
        metric_meter:   MetricMeter,
        loss_penalizer: LossPenalizer,
    ):
        device = self.device
        model = self.model
        model.train()
        meter = metric_meter
        penalizer = loss_penalizer
        meter.reset()
        penalizer.reset()

        desc = "Model Train"
        batch = len(databatchs)
        step = 0
        yield { "type": "init", "data" : {
            "desc" : desc,
            "batch": batch,
            },}

        for x_src, x_tgt in databatchs:
            x_src = x_src.to(device)
            x_tgt = x_tgt.to(device)
            optimizer.zero_grad()
            logits, attn_weights  = model(x_src, x_tgt[:, :-1])
            loss, grad_loss       = meter.calc_loss(logits, x_tgt[:, 1:])
            ids_pen, grad_ids_pen = penalizer.calc_penalty(logits, x_tgt[:, 1:])
            grad_loss = grad_loss + grad_ids_pen
            grad_loss.backward()
            optimizer.step()
            step += 1
            yield { "type": "update", "data" : {
                "step" : step,
                "loss" : loss,
                "ids_pen" : ids_pen,
                },}

        loss    = meter.loss
        ids_pen = penalizer.ids_penalty

        # return loss
        yield { "type" : "final", "data" : {
            "loss"    : loss,
            "ids_pen" : ids_pen,
            },}

    @torch.no_grad()
    def eval_model( self, databatchs,
        scheduler:      torch.optim.lr_scheduler,
        metricmeter:    MetricMeter,
    ):
        device = self.device
        model = self.model
        model.eval()
        meter = metricmeter
        meter.reset()

        desc = "Model Eval "
        batch = len(databatchs)
        step = 0
        yield { "type": "init", "data" : {
            "desc" : desc,
            "batch": batch,
            },}

        for x_src, x_tgt in databatchs:
            x_src = x_src.to(device)
            x_tgt = x_tgt.to(device)
            logits, _ = model(x_src, x_tgt[:, :-1])
            loss, _ = meter.calc_loss(logits, x_tgt[:, 1:])
            bleu, _ = meter.calc_bleu(logits, x_tgt[:, 1:])
            step += 1
            yield { "type": "update", "data" : {
                "step" : step,
                "loss" : loss,
                "bleu" : bleu,
                },}

        loss = meter.loss
        bleu = meter.bleu
        scheduler.step(loss)

        # return loss, bleu
        yield { "type" : "final", "data" : {
            "loss" : loss,
            "bleu" : bleu,
            },}

    @torch.no_grad()
    def infer_model(self, text, seq_len,
        pad_id, unk_id, sos_id, eos_id,
        tokenizer_src,
        tokenizer_tgt,
    ):
        device = self.device
        model  = self.model
        model.eval()

        src_ids = tokenizer_src.encode(text).ids
        src_ids = [sos_id] + src_ids + [eos_id]
        src_ids = src_ids[:seq_len]
        while len(src_ids) < seq_len : src_ids.append(pad_id)
        tgt_ids = [sos_id]

        x_src = torch.tensor([src_ids], dtype=torch.long).to(device)
        x_tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
        text = ""

        for _ in range(seq_len-1):
            logits, _ = model(x_src, x_tgt)
            x_next = logits.argmax(dim=-1)[:, -1].unsqueeze(1)
            x_tgt = torch.cat([x_tgt, x_next], dim=1)
            next_id = x_next[0].item()
            next_word = tokenizer_tgt.decode([next_id])
            text += "".join(next_word)
            if next_id == eos_id : break
            yield next_word

        # return text


