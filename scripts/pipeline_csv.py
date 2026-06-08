from process_csv import TextProcessor
from tokenize_txt import TextTokenizer


def main():

    input_path = "./data/en2cn_simple.csv"
    col_names = ["en","cn"]
    processor = TextProcessor(input_path, col_names, min_len=2, max_len=128)
    processor.run()

    input_path = "./data/prosess_text.csv"
    tokenizer = TextTokenizer(input_path,max_len=128)
    tokenizer.run()


if __name__ == "__main__":
    main()


