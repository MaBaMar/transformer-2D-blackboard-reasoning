"""Script to play around with blackboard state generation"""

from projectlib.my_datasets._blackboard_operands import *
from projectlib.my_datasets.blackboards import BASE_GEN_SPEC, BasicOpBlackboardDataset, BasicOpBlackboardIterator, BlackboardSpec, BB_PAD_TOKEN, BB_ROW_SEP_TOKEN
from transformers import AutoTokenizer

import numpy as np
import torch

# TODO: after finalizing tokenization and dataset layout, move to utils.py
def print_board(board):
    rows = board.replace(' ', '').replace(BB_PAD_TOKEN, ' ').split(BB_ROW_SEP_TOKEN)[:-1]
    print("-" * (len(rows[0]) + 2))
    for row in rows:
        print(f"|{row}|")
    print("-" * (len(rows[0]) + 2))

if __name__ == "__main__":

    # perform and inspect a demo operation
    x = np.array([9, 9, 0, 9])
    y = np.array([3, 9, 2, 9])

    spec = BlackboardSpec(15, 15, True, Addition())
    tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)

    b = BasicOpBlackboardIterator(x, y, spec)
    for board in b:
        print_board(board)

    gen_spec = BASE_GEN_SPEC
    gen_spec.size = 1
    dataset = BasicOpBlackboardDataset(regenerate=True, blackboard_spec=BlackboardSpec(5, 15, True, Addition()))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'pad_token': BB_PAD_TOKEN, 'sep_token': BB_ROW_SEP_TOKEN})

    # visually compare tokenized to expected blackboard
    torch.set_printoptions(linewidth=200)

    input_str = dataset[0]['input']
    token_seq = tokenizer(dataset[0]['input'])['input_ids']

    assert(len(token_seq) == len(input_str.split(" "))+2)
    print(torch.tensor(token_seq[1:-1]).view((5,16))) #omit sequence start and sequence end
    print_board(dataset[0]['input'])
    print_board(dataset[0]['label'])

    # some tokenizers split symbols into multiple tokens. We don't want that. Maybe think about it in next meeting.
    t = AutoTokenizer.from_pretrained("t5-small")
    print(t.encode("-", add_special_tokens=False))
