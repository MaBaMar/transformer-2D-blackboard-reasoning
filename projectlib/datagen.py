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


#
#   Functions that generate train/eval datasets of each type
#


def generate_addition(digits: int, low: int, high: int):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    AdditionDataset(
        path=EVAL_PATH_BASE.format(f"addition_{digits}"),
        tokenizer=tokenizer,
        train=False,
        regenerate=REGENERATE,
        generation_spec=GenerationSpec(EVAL_SIZE, low, high),
        operand=OPERATION,
    )


def generate_scratchpad(digits: int, low: int, high: int):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    ScratchpadDataset(
        path=EVAL_PATH_BASE.format(f"scratchpad_{digits}"),
        tokenizer=tokenizer,
        train=False,
        regenerate=REGENERATE,
        generation_spec=GenerationSpec(EVAL_SIZE, low, high),
        operand=OPERATION,
    )


def generate_blackboard(digits: int, low: int, high: int):
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


def main(digits: int):
    low = 10 ** (digits - 1)
    high = 10 ** digits

    print(f"Generating datasets with {digits}-digit numbers in the range [{low}, {high}) ...")

    generate_addition(digits, low, high)
    generate_scratchpad(digits, low, high)
    generate_blackboard(digits, low, high)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits", type=int, help="Amount of digits each number has.")

    args = parser.parse_args()

    main(digits=args.digits)
