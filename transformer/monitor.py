import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


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


