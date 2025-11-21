import os, sys, math, torch
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


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
        print(
            f"pad_id={pad_id}", "|",
            f"unk_id={unk_id}", "|",
            f"sos_id={sos_id}", "|",
            f"eos_id={eos_id}", "|",
            )
        return pad_id, unk_id, sos_id, eos_id

    def get_vocab_size(self, tokenizer):
        vocab_size = tokenizer.get_vocab_size()
        print(f"vocab size : {vocab_size}")
        # print("-" * 40)
        return vocab_size

    def reverse_vocab(self, tokenizer):
        vocab = tokenizer.get_vocab()
        new_vocab = {int(v): str(k) for k, v in vocab.items()}
        print(f"Vocab reversed.")
        print("-" * 40)
        return new_vocab

    def batch_data(self,
            path_tokenid_src,
            path_tokenid_tgt,
            batch_size, data_shuffle,
            ):
        x_src = torch.load(path_tokenid_src)
        x_tgt = torch.load(path_tokenid_tgt)
        print(f"src x shape : {tuple(x_src.shape)}")
        print(f"tgt x shape : {tuple(x_tgt.shape)}")
        # print("-" * 40)
        # truncated_len = 128
        # x_src = x_src[:, :truncated_len]
        # x_tgt = x_tgt[:, :truncated_len]
        dataset = TensorDataset(x_src, x_tgt)
        del x_src, x_tgt
        databatchs = DataLoader(dataset,
            batch_size=batch_size, shuffle=data_shuffle, drop_last=True)
        print(f"Data batched.")
        print("-" * 40)
        return databatchs

    def save_log(self, log_content, save_name="model.log"):
        data_dir = self.data_dir
        save_path = os.path.join(data_dir, save_name)
        current_time = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
        log_line = f"[{current_time}]{log_content}\n"
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(log_line)
        # print(f"{log_line.strip()}")
        print(f"Model Log saved, {save_path}")

    def save_model_weight(self, model, save_name="model_weight.pth"):
        data_dir = self.data_dir
        save_path = os.path.join(data_dir, save_name)
        torch.save(model.state_dict(), save_path)
        print(f"Model Weights saved, {save_path}")
        # print("=" * 40)

    def check_tokenid(self, path_tokenid, id2word):
        x = torch.load(path_tokenid)
        ids = x[100].tolist()
        text_gen = [id2word[id] for id in ids]
        text_gen = " ".join(text_gen)
        # print(ids)
        print(text_gen)

class LossMeter():
    def __init__(self, pad_id, label_smoothing):
        self.epoch_avg_loss = 0.0
        self.epoch_all_loss = 0.0
        self.epoch_id_num   = 0.0
        self.batch_avg_loss = 0.0
        self.batch_all_loss = 0.0
        self.batch_id_num   = 0.0
        self.pad_id = pad_id
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=pad_id, label_smoothing=label_smoothing,
            reduction="mean" )

    def calc_batch_loss(self, logits, x_tgt):
        pad_id = self.pad_id
        x_tgt_flat = x_tgt.reshape(-1)
        grad_loss = self.criterion(logits.reshape(-1, logits.size(-1)), x_tgt_flat)
        self.batch_avg_loss = grad_loss.item()
        self.batch_id_num   = (x_tgt != pad_id).sum().item()
        self.batch_all_loss = self.batch_avg_loss * self.batch_id_num
        return self.batch_avg_loss, grad_loss

    def calc_epoch_loss(self):
        self.epoch_all_loss += self.batch_all_loss
        self.epoch_id_num   += self.batch_id_num
        self.epoch_avg_loss  = self.epoch_all_loss / self.epoch_id_num
        return self.epoch_avg_loss

    def reset(self):
        self.epoch_all_loss = 0.0
        self.epoch_avg_loss = 0.0
        self.epoch_id_num   = 0.0

class BleuMeter():
    def __init__(self, pad_id, unk_id, sos_id, eos_id, id2word_tgt, weights):
        self.epoch_cor_bleu = 0.0
        self.epoch_all_ref  = []
        self.epoch_all_cand = []
        self.batch_cor_bleu = 0.0
        self.batch_all_ref  = []
        self.batch_all_cand = []
        self.pad_id  = pad_id
        self.unk_id  = unk_id
        self.sos_id  = sos_id
        self.eos_id  = eos_id
        self.id2word = id2word_tgt
        self.weights = weights
        self.smooth_fn = SmoothingFunction().method1

    def calc_batch_bleu(self, logits, x_tgt):
        pad_id  = self.pad_id
        unk_id  = self.unk_id
        sos_id  = self.sos_id
        eos_id  = self.eos_id
        id2word   = self.id2word
        weights   = self.weights
        smooth_fn = self.smooth_fn
        unk_token = id2word.get(unk_id)
        x_pred = logits.argmax(dim=-1)
        batch_all_bleu = []
        self.batch_all_ref  = []
        self.batch_all_cand = []
        for pred_ids, tgt_ids in zip(x_pred, x_tgt):
            valid_tgt_ids  = [id.item() for id in tgt_ids  if id.item() not in [pad_id,sos_id,eos_id]]
            valid_pred_ids = [id.item() for id in pred_ids if id.item() not in [pad_id,sos_id,eos_id]]
            true_text = "".join([id2word.get(id, unk_token) for id in valid_tgt_ids])
            pred_text = "".join([id2word.get(id, unk_token) for id in valid_pred_ids])
            ref_words  = list(true_text) if true_text else [""]
            cand_words = list(pred_text) if pred_text else [""]
            # ref_words  = true_text.split()
            # cand_words = pred_text.split()
            bleu = sentence_bleu([ref_words], cand_words, weights=weights, smoothing_function=smooth_fn)
            batch_all_bleu.append(bleu)
            self.batch_all_ref.append([ref_words])
            self.batch_all_cand.append(cand_words)
        sent_bleu = sum(batch_all_bleu) / len(batch_all_bleu)
        self.batch_cor_bleu = corpus_bleu(self.batch_all_ref, self.batch_all_cand, weights=weights, smoothing_function=smooth_fn)
        return self.batch_cor_bleu, sent_bleu

    def calc_epoch_bleu(self):
        weights   = self.weights
        smooth_fn = self.smooth_fn
        self.epoch_all_ref.extend( self.batch_all_ref)
        self.epoch_all_cand.extend(self.batch_all_cand)
        self.epoch_cor_bleu = corpus_bleu(self.epoch_all_ref, self.epoch_all_cand, weights=weights, smoothing_function=smooth_fn)
        return self.epoch_cor_bleu

    def reset(self):
        self.epoch_cor_bleu = 0.0
        self.epoch_all_ref  = []
        self.epoch_all_cand = []

class MetricMeter():
    def __init__(self, pad_id, unk_id, sos_id, eos_id,
        label_smoothing=0.0,
        id2word_tgt={},
        bleu_weights=[0.25, 0.25, 0.25, 0.25],
    ):
        self.loss_meter = LossMeter(pad_id, label_smoothing)
        self.bleu_meter = BleuMeter(pad_id, unk_id, sos_id, eos_id, id2word_tgt, bleu_weights)
        self.loss = 0.0
        self.bleu = 0.0

    def calc_loss(self, logits, x_tgt):
        batch_avg_loss, grad_loss = self.loss_meter.calc_batch_loss(logits, x_tgt)
        self.loss = self.loss_meter.calc_epoch_loss()
        return batch_avg_loss, grad_loss

    def calc_bleu(self, logits, x_tgt):
        batch_cor_bleu, sent_bleu = self.bleu_meter.calc_batch_bleu(logits, x_tgt)
        self.bleu = self.bleu_meter.calc_epoch_bleu()
        return batch_cor_bleu, sent_bleu

    def reset(self,):
        self.loss = 0.0
        self.bleu = 0.0
        self.loss_meter.reset()
        self.bleu_meter.reset()

class IdsPenalizer():
    def __init__(self, pad_id, penalty_ids, penalty_weight):
        self.epoch_avg_penalty = 0.0
        self.epoch_all_penalty = 0.0
        self.epoch_id_num      = 0.0
        self.batch_avg_penalty = 0.0
        self.batch_all_penalty = 0.0
        self.batch_id_num      = 0.0
        self.pad_id         = pad_id
        self.penalty_ids    = penalty_ids
        self.penalty_weight = penalty_weight

    def calc_batch_penalty(self, logits, x_tgt):
        pad_id         = self.pad_id
        penalty_ids    = self.penalty_ids
        penalty_weight = self.penalty_weight
        # penalty_mask = ~x_tgt.isin(penalty_ids) & (x_tgt != pad_id)
        penalty_mask = ~torch.any(torch.stack([x_tgt == id for id in penalty_ids], dim=-1), dim=-1)
        penalty_mask = penalty_mask & (x_tgt != pad_id)
        penalty_mask = penalty_mask.float()
        all_penalty = torch.tensor(0.0, device=logits.device)
        for id in penalty_ids:
            id_prob = torch.softmax(logits, dim=-1)[..., id]
            id_penalty = (id_prob * penalty_mask).sum()
            all_penalty += id_penalty
        id_num = (x_tgt != pad_id).sum().item()
        grad_penalty = all_penalty / id_num * penalty_weight * 100
        self.batch_id_num      = id_num
        self.batch_all_penalty = all_penalty.item()
        self.batch_avg_penalty = grad_penalty.item()
        return self.batch_avg_penalty, grad_penalty

    def calc_epoch_penalty(self):
        self.epoch_all_penalty += self.batch_all_penalty
        self.epoch_id_num      += self.batch_id_num
        self.epoch_avg_penalty  = self.epoch_all_penalty / self.epoch_id_num
        return self.epoch_avg_penalty

    def reset(self):
        self.epoch_avg_penalty = 0.0
        self.epoch_all_penalty = 0.0
        self.epoch_id_num      = 0.0

class LossPenalizer():
    def __init__(self, pad_id,
        penalty_ids=[],
        penalty_weight=0.10,
        # entropy_enc_weight=0.01,
        # entropy_dec_weight=0.01,
    ):
        self.ids_penalizer = IdsPenalizer(pad_id, penalty_ids, penalty_weight)
        self.ids_penalty = 0.0
        # self.entropy_meter = AttnEntropyMeter(entropy_enc_weight, entropy_dec_weight)
        # self.entropy = 0.0

    def calc_penalty(self, logits, x_tgt):
        batch_avg_penalty, grad_penalty = self.ids_penalizer.calc_batch_penalty(logits, x_tgt)
        self.ids_penalty = self.ids_penalizer.calc_epoch_penalty()
        return batch_avg_penalty, grad_penalty

    # def calc_entropy(self, attn_weights):
        # entropy = self.entropy_meter.calc(attn_weights)
        # self.entropy = self.entropy_meter.entropy
        # return entropy

    def reset(self,):
        self.ids_penalty = 0.0
        # self.entropy = 0.0
        self.ids_penalizer.reset()
        # self.entropy_meter.reset()

class EarlyStopper:
    def __init__(self,
        metric_name,
        patience=2,
        delta=0.01,
    ):
        self.metric = {"name":metric_name, "patience":patience, "delta":delta}
        self.metric.update({"counter": 0, "best_val": None})

    def track_metric(self, metric_val):
        metric = self.metric
        if metric["best_val"] is None:
            metric["best_val"] = metric_val
            return
        match metric["name"]:
            case "loss":
                is_worse = metric_val >= metric["best_val"] + metric["delta"]
            case "bleu":
                is_worse = metric_val <= metric["best_val"] - metric["delta"]
        if is_worse:
            metric["counter"] += 1
        else:
            metric["best_val"] = metric_val
            metric["counter"] = 0
        if metric["counter"] >= metric["patience"] :
            print(f"Metric {metric['name']} activate stop.")
            exit()

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

# class AttnPenalty():
    # def __init__(self, enc_weight, dec_weight):
    #     self.entropy     = 0.0
    #     self.all_entropy = 0.0
    #     self.sample_num  = 0.0
    #     self.enc_weight = enc_weight
    #     self.dec_weight = dec_weight
    #
    # def calc_ent(self, attn_weight, eps=1e-10):
    #     attn_weight = attn_weight.detach()
    #     weight = attn_weight.clamp(min=eps)
    #     entropy = -torch.sum(weight * torch.log(weight), dim=-1)
    #     seq_len = weight.shape[-1]
    #     max_entropy = math.log2(seq_len)
    #     entropy = entropy / max_entropy
    #     entropy = entropy.mean()
    #     return entropy
    #
    # def calc(self, attn_weights):
    #     enc_weight = self.enc_weight
    #     dec_weight = self.dec_weight
    #     enc_entropy = []
    #     dec_entropy = []
    #     for layer_w in attn_weights["enc"]:
    #         for head_w in layer_w:
    #             ent = self.calc_ent(head_w)
    #             enc_entropy.append(ent)
    #     for layer_w in attn_weights["dec"]:
    #         for head_w in layer_w:
    #             ent = self.calc_ent(head_w)
    #             dec_entropy.append(ent)
    #     enc_avg_entropy = sum(enc_entropy) / len(enc_entropy) * enc_weight * 100
    #     dec_avg_entropy = sum(dec_entropy) / len(dec_entropy) * dec_weight * 100
    #     batch_avg_entropy = (enc_avg_entropy + dec_avg_entropy) / 2
    #     self.all_entropy += batch_avg_entropy
    #     self.sample_num  += 1
    #     self.entropy      = self.all_entropy / self.sample_num
    #     return batch_avg_entropy
    #
    # def reset(self):
    #     self.entropy     = 0.0
    #     self.all_entropy = 0.0
    #     self.sample_num  = 0.0


