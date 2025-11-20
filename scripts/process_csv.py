from typing import Optional, Tuple, List
import os, re
import pandas as pd


class TextProcessor:
    def __init__(
        self,
        input_path:  str,
        col_names: List[str],
        min_len: int =  2,
        max_len: int = 128,
        separator="\t",
    ):
        self.input_path  = input_path
        self.col_names = col_names
        self.min_len = min_len
        self.max_len = max_len
        self.df = None
        self.sep = separator

    def _load_text(self):
        try:
            self.df = pd.read_csv(self.input_path, sep=self.sep, encoding="utf-8-sig", on_bad_lines="skip")
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.input_path, sep=self.sep, encoding="gbk", on_bad_lines="skip")
        missing_cols = [col for col in self.col_names if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV中缺少指定的文本列：{missing_cols}")
        # self.df = self.df.dropna(subset=self.txt_cols, how="all")
        print(f"文本加载完成，原始文本{len(self.df)}行")

    def _clean_cell(self):
        if self.df is None:
            raise ValueError("请先调用_load_text加载数据")
        self.df = self.df.dropna(subset=self.col_names, how="any")
        def clean_single_cell(text):
            if pd.isna(text): return None
            text = re.sub(r'[\n\r]', '', str(text).strip())
            text = re.sub(r"(?<![，。！？',.!?a-zA-Z])\s+(?![，。！？,.!?a-zA-Z])", "", text)
            text = re.sub(r"[^，。！？',.!?0-9a-zA-Z\u4e00-\u9fa5\s]", "", text)
            return text if text.strip() else None
        for col in self.col_names:
            self.df[col] = self.df[col].apply(clean_single_cell)
        self.df = self.df.dropna(subset=self.col_names, how="any")
        print(f"文本清洗完成，剩余文本{len(self.df)}行")

    def _filter_row(self):
        if self.df is None:
            raise ValueError("请先调用_load_text加载数据")
        def check_length(text):
            return self.min_len <= len(str(text)) <= self.max_len
        for col in self.col_names:
            self.df = self.df[self.df[col].apply(check_length)]
        print(f"文本去行完成，剩余文本{len(self.df)}行")

    def _filter_col(self):
        if self.df is None:
            raise ValueError("请先调用_load_text加载数据")
        new_col_names = {}
        new_col_names[self.col_names[0]] = "src_text"
        new_col_names[self.col_names[1]] = "tgt_text"
        self.df = self.df[self.col_names]
        self.df = self.df.rename(columns=new_col_names)
        print(f"文本去列完成，剩余文本{len(self.df)}行")

    def _save_text(self):
        if self.df is None:
            raise ValueError("请先调用_load_text加载数据")
        dirname = os.path.dirname(self.input_path)
        save_path = os.path.join(dirname, "prosess_text.csv")
        self.df.to_csv(save_path, sep=self.sep, index=False, encoding="utf-8-sig")
        print(f"文本已经保存，{save_path}")

    def run(self):
        self._load_text()
        self._clean_cell()
        self._filter_row()
        self._filter_col()
        self._save_text()
        print("=" * 40)
        print("文本数据处理完成！")


def main():
    input_path = "../data/en2cn_simple.csv"
    col_names = ["en","cn"]
    processor = TextProcessor(input_path, col_names, min_len=2, max_len=128)
    processor.run()


if __name__ == "__main__":
    main()


