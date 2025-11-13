from dataclasses import dataclass
from typing import override
import torch
import numpy as np

from torch.types import Number
from transformers import AutoTokenizer

from projectlib.my_datasets.base import GeneratedDataset, GenerationSpec
from projectlib.my_datasets._blackboard_operands import Addition, CarryOperation

@dataclass
class BlackboardSpec:
    """Class that specifies the parameters for the blackboard generation."""
    height: int
    width: int
    randomize_position: bool
    operation: CarryOperation

# blackboard configuration
EVAL_PATH_BASE = "datasets/bb_{}_eval.pt"
TRAIN_PATH_BASE = "datasets/bb_{}_train.pt"
BASE_GEN_SPEC = GenerationSpec(10, 10, 20)
BASE_BLACKBOARD_SPEC = BlackboardSpec(5, 15, False, Addition())

# tokenization configuration
BB_ROW_SEP_TOKEN = "[SEP]"
BB_PAD_TOKEN     = "[PAD]"

class BasicOpBlackboardIterator:
    def __init__(
        self,
        op1: np.ndarray,
        op2: np.ndarray,
        spec: BlackboardSpec = BASE_BLACKBOARD_SPEC,
    ):
        """
        Iterator that generates a sequence of blackboard states for the specified operations. It also takes care of board position randomization
        and ensures that the blackboard states fit the requested dimensions.

        Args:
            op1 (np.ndarray): The first operand encoded as digit array.
            op2 (np.ndarray): The second operand encoded as digit array.
            spec (BlackboardSpec): The blackboard specification.
        """

        if len(op1) != len(op2):
            raise ValueError("Operation arrays must be of equal length. Use zero padding")
        if spec.width < len(op1) + 2 or spec.height < 5:
            raise ValueError(f"Generated blackboard states cannot fit the requested dimensions of {spec.height} x {spec.width}. Input numbers require a size of at least 5 x {len(op1) + 2}")

        self.spec = spec
        self.op1 = op1
        self.op2 = op2
        self.oplen = len(op1)
        self.step = 0
        self.last_carry = 0

        self.curr_bb_state = [
            [BB_PAD_TOKEN]+(self.oplen+1)*["_"],
            2*[BB_PAD_TOKEN] + [str(x) for x in op1],
            [str(spec.operation), BB_PAD_TOKEN] + [str(x) for x in op2],
            (self.oplen+2) * ["-"],
            [BB_PAD_TOKEN]+["_"]*(self.oplen+1)
        ]

        self.random_x_start = 0
        self.random_y_start = 0

        if self.spec.randomize_position:
            r_max_x = self.spec.width - self.oplen - 2
            r_max_y = self.spec.height - 5
            if r_max_x > 0:
                self.random_x_start = np.random.randint(0, r_max_x)
            if r_max_y > 0:
                self.random_y_start = np.random.randint(0, r_max_y)


    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):

        if self.step == self.oplen + 2:
            raise StopIteration

        retval = self._position_and_flatten()
        self.step += 1

        if self.step <= self.oplen:
            res, carry = self.spec.operation.step(self.op1[-self.step], self.op2[-self.step], self.last_carry)
            self.curr_bb_state[0][-(self.step+1)] = str(carry)[0]
            self.curr_bb_state[4][-(self.step)] = str(res)[0]
            self.last_carry = carry

        if self.step == self.oplen + 1:
            self.curr_bb_state[4][-(self.step)] = self.spec.operation.finalize_carry(self.last_carry)

        return retval

    def _int_to_list(self, num: int):
        return [int(digit) for digit in str(num)]

    def _position_and_flatten(self):
        """Position the blackboard and pack the current state of the blackboard into a encoded string suitable for tokenization."""

        # create a 2d grid of size self.spec.height x self.spec.width
        x_pad_start = self.random_x_start * [BB_PAD_TOKEN]  # padding in front of each row
        x_pad_end = (self.spec.width - self.random_x_start - self.oplen - 2) * [BB_PAD_TOKEN] + [BB_ROW_SEP_TOKEN] # padding at the end of each row

        ypadstr = self.spec.width * [BB_PAD_TOKEN] + [BB_ROW_SEP_TOKEN]    # this is just an empty row
        y_pad_start = self.random_y_start * [ypadstr] # empty rows before blackboard data starts
        y_pad_end = (self.spec.height - self.random_y_start - 5) * [ypadstr] # empty rows after blackboard data ends

        grid = (
            y_pad_start +
            [x_pad_start + x + x_pad_end for x in self.curr_bb_state] +
            y_pad_end
        )

        # linearize the grid into a single string, separate all tokens with a space so the tokenizer can separate them
        grid_str = ' '.join([' '.join(row) for row in grid])
        return grid_str

class BasicOpBlackboardDataset(GeneratedDataset):
    def __init__(
        self,
        path: str = None,
        tokenizer: AutoTokenizer = None,
        train: bool = True,
        regenerate: bool = False,
        generation_spec: GenerationSpec = BASE_GEN_SPEC,
        blackboard_spec: BlackboardSpec = BASE_BLACKBOARD_SPEC,
    ):
        path = path or (TRAIN_PATH_BASE.format(blackboard_spec.operation.get_name()) if train else EVAL_PATH_BASE.format(blackboard_spec.operation.get_name()))

        self.specs = blackboard_spec

        # generate data
        super().__init__(
            path,
            tokenizer,
            regenerate,
            generation_spec,
        )

        # preapare tokenizer, make sure it knows the sep token
        if(self.tokenizer is not None):
            self.tokenizer.add_special_tokens({'sep_token': BB_ROW_SEP_TOKEN, 'pad_token': BB_PAD_TOKEN})

    @override
    def __generate__(self, spec: GenerationSpec):
        inputs = []
        labels = []

        for _ in range(spec.size):
            # size is interpreted as the number of blackboard computation chains to generate
            a = torch.randint(spec.low, spec.high, (1,)).item()
            b = torch.randint(spec.low, spec.high, (1,)).item()

            if(self.specs.operation.get_name() == 'subtraction') and a < b:
                a, b = b, a

            input_states, output_states = self._generate_blackboard_pairs(a, b)
            inputs += input_states
            labels += output_states

        print("Generated", spec.size, "blackboard chains consisting of", len(inputs)+1, "blackboards")

        return inputs, labels

    def _generate_blackboard_pairs(self, a: Number, b: Number):

        # perform hand addition or subtraction and store the digit steps in lists
        max_input_length = max(np.floor(np.log10(a)).astype(np.int32), np.floor(np.log10(b)).astype(np.int32))

        digits_a = np.empty(max_input_length, dtype=int)
        digits_b = np.empty(max_input_length, dtype=int)
        for i in range(max_input_length):
            digits_a[max_input_length - i - 1] = a % 10
            a //= 10
            digits_b[max_input_length - i - 1] = b % 10
            b //= 10

        blackboards = [*BasicOpBlackboardIterator(digits_a, digits_b, self.specs)]

        return blackboards[:-1], blackboards[1:]

    def __getitem__(self, idx: int):
        # TODO: implement custom __getitem__ method which does correct tokenization
        # Note from Marco: I will update this once Lino communicates the model's requirements
        return super().__getitem__(idx)
