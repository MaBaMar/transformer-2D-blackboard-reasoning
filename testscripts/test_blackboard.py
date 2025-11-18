# ------------------------------------------------------------
# test_blackboard.py
#
# Test for blackboard.py and associated modules.
#
# Run with:
#  - python -m testscripts.test_blackboard
#  - run from the project root
# ------------------------------------------------------------

import os
import pytest
import torch

from projectlib.my_datasets._blackboard_operands import Addition, Subtraction
from projectlib.my_datasets.blackboards import (
    BlackboardSpec,
    GenerationSpec,
    TokenizedBlackboardDataset,
    bb_prettyprint,
    bb_datasample_prettyprint
)

# --- 1. Fixtures for Setup and Shared Data ---

@pytest.fixture(scope="session")
def setup_data_tmp_path(tmp_path_factory):
    data_tmp_path = tmp_path_factory.mktemp("datacache")
    return str(data_tmp_path)

@pytest.fixture(scope="session")
def gen_spec():
    return GenerationSpec(size=1, low=10, high=99)

@pytest.fixture(scope="session")
def test_chain_data_add():
    return [
        [['<BOS>', '<EMPTY>', '6', '8', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['+', '<EMPTY>', '7', '9', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '_', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '-', '-', '-', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '_', '_', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>']],
        [['<BOS>', '<EMPTY>', '6', '8', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['+', '<EMPTY>', '7', '9', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '1', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '-', '-', '-', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '_', '7', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>']],
        [['<BOS>', '<EMPTY>', '6', '8', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['+', '<EMPTY>', '7', '9', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '1', '1', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '-', '-', '-', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '4', '7', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>']],
        [['<BOS>', '<EMPTY>', '6', '8', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['+', '<EMPTY>', '7', '9', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '1', '1', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '-', '-', '-', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '1', '4', '7', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>']],
        [['<BOS>', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'], ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'], ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'], ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'], ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']],
    ]

@pytest.fixture(scope="session")
def test_chain_data_sub():
    return [
        [['<BOS>', '<EMPTY>', '7', '9', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '<EMPTY>', '6', '8', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '_', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '-', '-', '-', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '_', '_', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>']],
        [['<BOS>', '<EMPTY>', '7', '9', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '<EMPTY>', '6', '8', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '0', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '-', '-', '-', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '_', '1', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>']],
        [['<BOS>', '<EMPTY>', '7', '9', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '<EMPTY>', '6', '8', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '0', '0', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '-', '-', '-', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '_', '1', '1', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>']],
        [['<BOS>', '<EMPTY>', '7', '9', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '<EMPTY>', '6', '8', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '0', '0', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['-', '-', '-', '-', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>'], ['<EMPTY>', '0', '1', '1', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>', '<EMPTY>']],
        [['<BOS>', '<EOS>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'], ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'], ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'], ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'], ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']],
    ]

# --- 2. The Pytest Function ---

def test_addition_board_sequence(setup_data_tmp_path, gen_spec, test_chain_data_add):
    """
    Tests the sequence of board states generated by the TokenizedBlackboardDataset
    for a simple addition problem.
    """
    bb_spec = BlackboardSpec(5, 10, False, Addition())
    data_path = os.path.join(setup_data_tmp_path, "bb_addition.pl")

    dataset = TokenizedBlackboardDataset(
        path=data_path,
        regenerate=True, # Force regeneration for reliable testing
        generation_spec=gen_spec,
        blackboard_spec=bb_spec,
    )

    for i, (x, y) in enumerate(dataset):
        t_x = dataset.bb_2D_tokenizer.encode(test_chain_data_add[i])
        t_y = dataset.bb_2D_tokenizer.encode(test_chain_data_add[i+1])

        assert torch.equal(x['tokens'], t_x), f"Failed at sample {i}: Expected\n{t_x},\ngot\n{x['tokens']}"
        assert torch.equal(y['tokens'], t_y), f"Failed at sample {i}: Expected\n{t_y},\ngot\n{y['tokens']}"


def test_subtraction_board_sequence(setup_data_tmp_path, gen_spec, test_chain_data_sub):
    """
    Tests the sequence of board states generated by the TokenizedBlackboardDataset
    for a simple subtraction problem.
    """
    bb_spec = BlackboardSpec(5, 10, False, Subtraction())
    data_path = os.path.join(setup_data_tmp_path, "bb_subtraction.pl")

    dataset = TokenizedBlackboardDataset(
        path=data_path,
        regenerate=True, # Force regeneration for reliable testing
        generation_spec=gen_spec,
        blackboard_spec=bb_spec,
    )

    for i, (x, y) in enumerate(dataset):
        t_x = dataset.bb_2D_tokenizer.encode(test_chain_data_sub[i])
        t_y = dataset.bb_2D_tokenizer.encode(test_chain_data_sub[i+1])

        assert torch.equal(x['tokens'], t_x), f"Failed at sample {i}: Expected\n{t_x},\ngot\n{x['tokens']}"
        assert torch.equal(y['tokens'], t_y), f"Failed at sample {i}: Expected\n{t_y},\ngot\n{y['tokens']}"


if __name__ == "__main__":
    # if we need debugging
    torch.set_printoptions(linewidth=250)
    pytest.main()

    # can add some visual testing of blackboard states
    dataset = TokenizedBlackboardDataset(
        path=os.path.join("bb_subtraction.pl"),
        regenerate=True, # Force regeneration for reliable testing
        generation_spec=GenerationSpec(size=1, low=10, high=99),
        blackboard_spec=BlackboardSpec(10, 10, True, Addition()),
    )

    for i, (x, y) in enumerate(dataset):
        print("X:")
        bb_prettyprint(x["tokens"])
        print("Y:")
        bb_prettyprint(y["tokens"])
        print(40*"=")
