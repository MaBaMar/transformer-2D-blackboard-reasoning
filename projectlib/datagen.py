"""
Function for dataset generation. Feel free to define new functions here and to use them in the code.
Please regularly push updated versions of the library, so others can use the same functionality.
"""
import argparse

from transformers import AutoTokenizer

from projectlib.my_datasets.base import GenerationSpec
from projectlib.my_datasets.additions import AdditionDataset
from projectlib.my_datasets.scratchpads import ScratchpadDataset
from projectlib.my_datasets.blackboards import BasicOpBlackboardDataset, BlackboardSpec, BB_PAD_TOKEN, BB_ROW_SEP_TOKEN
from projectlib.my_datasets._blackboard_operands import *



TRAIN_SIZE = 10000
EVAL_SIZE = 1000

OPERATION = "+"

REGENERATE = True

BB_HEIGHT = 32
BB_WIDTH = 32
BB_RANDOMIZE_POS = False
BB_OPERATION = Addition()

TRAIN_PATH_BASE = "datasets/{}_train.pt"
EVAL_PATH_BASE = "datasets/{}_eval.pt"

TOKENIZERS = {
    "None": None,
    "bert": "bert-base-uncased",
    "t5": "t5-small",
}


#
#   Functions that generate train/eval datasets of each type
#


def generate_addition(digits: int, low: int, high: int, tokenizer: AutoTokenizer=None):
    AdditionDataset(
        path=EVAL_PATH_BASE.format(f"addition_{digits}"),
        tokenizer=tokenizer,
        train=False,
        regenerate=REGENERATE,
        generation_spec=GenerationSpec(EVAL_SIZE, low, high),
        operand=OPERATION,
    )


def generate_scratchpad(digits: int, low: int, high: int, tokenizer: AutoTokenizer=None):
    ScratchpadDataset(
        path=EVAL_PATH_BASE.format(f"scratchpad_{digits}"),
        tokenizer=tokenizer,
        train=False,
        regenerate=REGENERATE,
        generation_spec=GenerationSpec(EVAL_SIZE, low, high),
        operand=OPERATION,
    )


def generate_blackboard(digits: int, low: int, high: int, tokenizer: AutoTokenizer=None):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'pad_token': BB_PAD_TOKEN, 'sep_token': BB_ROW_SEP_TOKEN})

    bb_spec = BlackboardSpec(
        height=BB_HEIGHT,
        width=BB_WIDTH,
        randomize_position=BB_RANDOMIZE_POS,
        operation=BB_OPERATION
    )

    name = f"bb_{bb_spec.operation.get_name()}_{digits}"

    BasicOpBlackboardDataset(
        path=TRAIN_PATH_BASE.format(name),
        tokenizer=tokenizer,
        regenerate=REGENERATE,
        train=True,
        generation_spec=GenerationSpec(TRAIN_SIZE, low, high),
        blackboard_spec=bb_spec,
    )

    BasicOpBlackboardDataset(
        path=EVAL_PATH_BASE.format(name),
        tokenizer=tokenizer,
        regenerate=REGENERATE,
        train=False,
        generation_spec=GenerationSpec(EVAL_SIZE, low, high),
        blackboard_spec=bb_spec,
    )


#
#   Generate all the datasets
#


def main(digits: int, tokenizer_name: str):
    low = 10 ** (digits - 1)
    high = 10 ** digits

    print(f"Generating datasets with {digits}-digit numbers in the range [{low}, {high}) using tokenizer: {tokenizer_name}.")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[tokenizer_name]) if tokenizer_name != "None" else None

    generate_addition(digits, low, high, tokenizer=tokenizer)
    generate_scratchpad(digits, low, high, tokenizer=tokenizer)
    generate_blackboard(digits, low, high, tokenizer=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits", type=int, help="Amount of digits each number has.")
    parser.add_argument("--tokenizer", type=str, default="None", help="Tokenizer to use (default: None).")

    args = parser.parse_args()

    main(digits=args.digits, tokenizer_name=args.tokenizer)
