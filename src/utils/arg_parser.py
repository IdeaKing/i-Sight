import argparse

parser = argparse.ArgumentParser(
    description="Run AMPL Training or Tests")
parser.add_argument(
    "--training-type",
    type=str,
    default="obd")
parser.add_argument(
    "--dataset-path",
    "-D",
    type=str,
    help="Path to dataset")
parser.add_argument(
    "--training-dir",
    "-T",
    type=str,
    help="Path to the training directory")
parser.add_argument(
    "--mixed-precision",
    type=bool,
    default=False)
parser.add_argument(
    "--serialize",
    "-S",
    type=bool,
    help="To serialize dataset",
    default=False)
parser.add_argument(
    "--finetune",
    "-F",
    type=bool,
    help="If finetune or not",
    default=False)

args = parser.parse_args()