import argparse
from .version import get_version_str


def parse_cli_args():
    """Parse command line arguments for pt-transformer framework"""
    parser = argparse.ArgumentParser(
        description="pt-transformer: A Lightweight Seq2Seq Training Framework Based on PyTorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

     # Version information (enhanced with status)
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s v{get_version_str()}",
        help="App Version"
        )

    # Core running mode (required parameter)
    parser.add_argument(
        "-m", "--mode",
        required=False,
        default="all",
        choices=["all", "train", "eval", "infer"],
        help="Running mode selection: all/train/eval/infer"
        )

    # Transformer architecture core parameters
    parser.add_argument(
        "--n-heads",
        type=int,
        help="Number of multi-head attention heads in Transformer"
        )
    parser.add_argument(
        "--enc-n-layers",
        type=int,
        help="Number of layers in Transformer encoder stack"
        )
    parser.add_argument(
        "--dec-n-layers",
        type=int,
        help="Number of layers in Transformer decoder stack"
        )

    # General hyperparameter
    parser.add_argument(
        "--epoch-total",
        type=int,
        help="Total number of training epochs for the whole training process"
        )

    return parser.parse_args()


