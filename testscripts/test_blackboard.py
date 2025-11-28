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
from torch.utils.data import DataLoader

from projectlib.my_datasets._blackboard_operands import Addition, Subtraction
from projectlib.my_datasets.blackboards import BB_EMPTY_TOKEN, BB_BOS_TOKEN, BB_EOS_TOKEN, BB_PAD_TOKEN, BB_FILL_NUM_TOKEN, BB_OPLINE_SEG_TOKEN
from projectlib.my_datasets.blackboards import (
    BlackboardSpec,
    GenerationSpec,
    TokenizedBlackboardDataset,
    bb_prettyprint,
)

from projectlib.my_datasets.collators import collate_blackboards, make_collator_with_args

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
        [[BB_BOS_TOKEN, BB_EMPTY_TOKEN, '6', '8', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], ['+', BB_EMPTY_TOKEN, '7', '9', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, BB_FILL_NUM_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, BB_FILL_NUM_TOKEN, BB_FILL_NUM_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN]],
        [[BB_BOS_TOKEN, BB_EMPTY_TOKEN, '6', '8', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], ['+', BB_EMPTY_TOKEN, '7', '9', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, '1', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, BB_FILL_NUM_TOKEN, '7', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN]],
        [[BB_BOS_TOKEN, BB_EMPTY_TOKEN, '6', '8', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], ['+', BB_EMPTY_TOKEN, '7', '9', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, '1', '1', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, '4', '7', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN]],
        [[BB_BOS_TOKEN, BB_EMPTY_TOKEN, '6', '8', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], ['+', BB_EMPTY_TOKEN, '7', '9', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, '1', '1', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, '1', '4', '7', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN]],
        [[BB_BOS_TOKEN, BB_EOS_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN], [BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN], [BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN], [BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN], [BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN]],
    ]

@pytest.fixture(scope="session")
def test_chain_data_sub():
    return [
        [[BB_BOS_TOKEN, BB_EMPTY_TOKEN, '7', '9', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], ['-', BB_EMPTY_TOKEN, '6', '8', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, BB_FILL_NUM_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, BB_FILL_NUM_TOKEN, BB_FILL_NUM_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN]],
        [[BB_BOS_TOKEN, BB_EMPTY_TOKEN, '7', '9', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], ['-', BB_EMPTY_TOKEN, '6', '8', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, '0', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, BB_FILL_NUM_TOKEN, '1', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN]],
        [[BB_BOS_TOKEN, BB_EMPTY_TOKEN, '7', '9', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], ['-', BB_EMPTY_TOKEN, '6', '8', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, '0', '0', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, BB_FILL_NUM_TOKEN, '1', '1', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN]],
        [[BB_BOS_TOKEN, BB_EMPTY_TOKEN, '7', '9', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], ['-', BB_EMPTY_TOKEN, '6', '8', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, '0', '0', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_OPLINE_SEG_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN], [BB_EMPTY_TOKEN, '0', '1', '1', BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN, BB_EMPTY_TOKEN]],
        [[BB_BOS_TOKEN, BB_EOS_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN], [BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN], [BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN], [BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN], [BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN, BB_PAD_TOKEN]],
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

    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (x,y) in enumerate(dl):
        t_x = dataset.bb_2D_tokenizer.encode(test_chain_data_add[i])
        t_y = dataset.bb_2D_tokenizer.encode(test_chain_data_add[i+1])

        assert torch.equal(x['tokens'][0], t_x), f"Failed at sample {i}: Expected\n{t_x},\ngot\n{x['tokens'][0]}"
        assert torch.equal(y['tokens'][0], t_y), f"Failed at sample {i}: Expected\n{t_y},\ngot\n{y['tokens'][0]}"


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

    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (x, y) in enumerate(dl):
        t_x = dataset.bb_2D_tokenizer.encode(test_chain_data_sub[i])
        t_y = dataset.bb_2D_tokenizer.encode(test_chain_data_sub[i+1])

        assert torch.equal(x['tokens'][0], t_x), f"Failed at sample {i}: Expected\n{t_x},\ngot\n{x['tokens'][0]}"
        assert torch.equal(y['tokens'][0], t_y), f"Failed at sample {i}: Expected\n{t_y},\ngot\n{y['tokens'][0]}"


if __name__ == "__main__":
    pytest.main()

    run_tmp_path = "tmp_bb_subtraction.pl"
    # can add some visual testing of blackboard states
    dataset = TokenizedBlackboardDataset(
        path=os.path.join(run_tmp_path),
        regenerate=True, # Force regeneration for reliable testing
        generation_spec=GenerationSpec(size=1, low=10, high=99),
        blackboard_spec=BlackboardSpec(10, 10, True, Addition()),
    )

    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (x,y) in enumerate(dl):
        print("X:")
        bb_prettyprint(x["tokens"][0])
        print("Y:")
        bb_prettyprint(y["tokens"][0])
        print(40*"=")

    # cleanup
    os.remove(run_tmp_path)
