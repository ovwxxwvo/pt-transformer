import os, json
from typing import Literal
import pandas as pd
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.utils.data import TensorDataset, random_split


class TextTokenizer():
    def __init__(
        self,
        input_path: str,
        max_len = 128,
        separator="\t",
    ):
        self.input_path = input_path
        self.max_len = max_len
        self.sep = separator
        self.df = None

    def _load_text(self):
        try:
            self.df = pd.read_csv(self.input_path, sep=self.sep, encoding="utf-8-sig", on_bad_lines="skip")
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.input_path, sep=self.sep, encoding="gbk", on_bad_lines="skip")
        print(f"Text data loaded，text line : {len(self.df)} .")

    def train_tokenizer(self, save_path, texts, vocab_size=10000):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            # special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
            # special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
            vocab_size=vocab_size,
            min_frequency=2,
            )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        print(f"Tokenizer already trained.")
        tokenizer.save(save_path)
        print(f"New tokenizer saved, {save_path}")
        print("-" * 40)
        return tokenizer

    def tokenize_text(self, tokenizer, text, seq_type:Literal["src","tgt"]):
        encode = tokenizer.encode(text)
        token_ids = encode.ids
        max_len = self.max_len
        sos_id = tokenizer.token_to_id("[SOS]")
        eos_id = tokenizer.token_to_id("[EOS]")
        if   seq_type == "src":
            token_ids = token_ids + [eos_id]
        elif seq_type == "tgt":
            token_ids = [sos_id] + token_ids + [eos_id]
        pad_id = tokenizer.token_to_id("[PAD]")
        if len(token_ids) < max_len:
            token_ids += [pad_id] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]
        # print(f"词元分词完成。")
        return torch.tensor(token_ids, dtype=torch.long)

    def split_tensor(self, src_tensor, tgt_tensor, eval_ratio=0.1, test_ratio=0.1, seed=42):
        total_num = src_tensor.shape[0]
        eval_num = int(total_num * eval_ratio)
        test_num = int(total_num * test_ratio)
        train_num = total_num - eval_num - test_num
        dataset = TensorDataset(src_tensor, tgt_tensor)
        train_set, eval_set, test_set = random_split(
            dataset, [train_num, eval_num, test_num],
            generator=torch.Generator().manual_seed(seed)
            )
        src_train_set = torch.stack([data[0] for data in train_set])
        tgt_train_set = torch.stack([data[1] for data in train_set])
        src_eval_set  = torch.stack([data[0] for data in eval_set])
        tgt_eval_set  = torch.stack([data[1] for data in eval_set])
        src_test_set  = torch.stack([data[0] for data in test_set])
        tgt_test_set  = torch.stack([data[1] for data in test_set])
        return (src_train_set, tgt_train_set), (src_eval_set, tgt_eval_set), (src_test_set, tgt_test_set)

    def save_vocab(self, save_path, tokenizer):
        vocab = tokenizer.get_vocab()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"Vocab saved, {save_path}")

    def save_tokenid(self, save_path, tensor):
        torch.save(tensor, save_path)
        print(f"Token id saved, {save_path}")


    def run(self):
        self._load_text()
        print("=" * 40)
    # path
        dirname = os.path.dirname(self.input_path)
        src_path_tokenizer = os.path.join(dirname, "tokenizer_src.json")
        tgt_path_tokenizer = os.path.join(dirname, "tokenizer_tgt.json")
        src_path_vocab     = os.path.join(dirname, "vocab_src.json")
        tgt_path_vocab     = os.path.join(dirname, "vocab_tgt.json")
        src_path_tokenid_train = os.path.join(dirname, "tokenid_src_train.pt")
        src_path_tokenid_eval  = os.path.join(dirname, "tokenid_src_eval.pt")
        src_path_tokenid_test  = os.path.join(dirname, "tokenid_src_test.pt" )
        tgt_path_tokenid_train = os.path.join(dirname, "tokenid_tgt_train.pt")
        tgt_path_tokenid_eval  = os.path.join(dirname, "tokenid_tgt_eval.pt")
        tgt_path_tokenid_test  = os.path.join(dirname, "tokenid_tgt_test.pt" )
    # tokenizer
        src_tokenizer = self.train_tokenizer(
            save_path=src_path_tokenizer,
            texts=self.df["src_text"].tolist(),
            vocab_size=10000
            )
        tgt_tokenizer = self.train_tokenizer(
            save_path=tgt_path_tokenizer,
            texts=self.df["tgt_text"].tolist(),
            vocab_size=20000
            )
    # tokenid
        src_tokenids = [self.tokenize_text(src_tokenizer, txt, seq_type="src") for txt in self.df["src_text"]]
        tgt_tokenids = [self.tokenize_text(tgt_tokenizer, txt, seq_type="tgt") for txt in self.df["tgt_text"]]
        src_tensor = torch.stack(src_tokenids)
        tgt_tensor = torch.stack(tgt_tokenids)
        (src_train_set, tgt_train_set), (src_eval_set, tgt_eval_set), (src_test_set, tgt_test_set) \
            = self.split_tensor(src_tensor, tgt_tensor)
        print("=" * 40)
    # save
        self.save_vocab(src_path_vocab, src_tokenizer)
        self.save_vocab(tgt_path_vocab, tgt_tokenizer)
        print("-" * 40)
        self.save_tokenid(src_path_tokenid_train, src_train_set )
        self.save_tokenid(src_path_tokenid_eval , src_eval_set  )
        self.save_tokenid(src_path_tokenid_test , src_test_set  )
        self.save_tokenid(tgt_path_tokenid_train, tgt_train_set )
        self.save_tokenid(tgt_path_tokenid_eval , tgt_eval_set  )
        self.save_tokenid(tgt_path_tokenid_test , tgt_test_set  )
        print("=" * 40)
        return src_tensor, tgt_tensor, src_tokenizer, tgt_tokenizer


def main():
    input_path = "../data/prosess_text.csv"
    tokenizer = TextTokenizer(input_path, max_len=128)
    tokenizer.run()


if __name__ == "__main__":
    main()


