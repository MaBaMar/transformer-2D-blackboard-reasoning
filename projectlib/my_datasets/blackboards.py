from dataclasses import dataclass
from typing import override
import torch
import numpy as np

from torch.types import Number
from transformers import AutoTokenizer

from my_datasets.base import GeneratedDataset, GenerationSpec
from my_datasets.blackboard_operands import Addition, CarryOperation

@dataclass
class BlackboardSpec:
    """Class that specifies the parameters for the blackboard generation."""
    height: int
    width: int
    randomize_position: bool
    operation: CarryOperation

EVAL_PATH_BASE = "datasets/{}_eval.pt"
TRAIN_PATH_BASE = "datasets/{}_train.pt"
BASE_GEN_SPEC = GenerationSpec(10, 10, 20)
BASE_BLACKBOARD_SPEC = BlackboardSpec(5, 15, False, Addition())


class BasicOpBlackboardIterator:
    def __init__(
        self,
        op1: np.ndarray,
        op2: np.ndarray,
        spec: BlackboardSpec = BASE_BLACKBOARD_SPEC,
    ):

        if len(op1) != len(op2):
            raise ValueError("Operation arrays must be of equal length. Use zero padding")
        if len(op1) + 2 < spec.width or spec.height < 5:
            raise ValueError(f"Generated blackboard states cannot fit the requested dimensions of {spec.height} x {spec.width}. Input numbers require a size of at least 5 x {len(op1) + 2}")

        self.spec = spec
        self.op1 = op1
        self.op2 = op2
        self.oplen = len(op1)
        self.step = 0

        self.curr_bb_state = [
            list(" _"+self.oplen*"_"),
            list("  " + "".join([(str(x) if x else '_') for x in op1])),
            list(spec.operation.get_name() + " " + "".join([(str(x) if x else '_') for x in op2])),
            list("--"+self.oplen*"-"),
            list(" _"+self.oplen*"_")
        ]

        if self.spec.randomize_position:
            self.random_x_start = np.random.randint(0, self.spec.width - self.oplen - 2)
            self.random_y_start = np.random.randint(0, self.spec.height - 5)
        else:
            self.random_x_start = 0
            self.random_y_start = 0

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):

        if self.step == self.oplen + 2:
            raise StopIteration

        retval = self._pack()
        self.step += 1

        if self.step <= self.oplen:
            res, carry = self.spec.operation.step(self.op1[-self.step], self.op2[-self.step])
            self.curr_bb_state[0][-(self.step+1)] = str(carry)[0]
            self.curr_bb_state[4][-(self.step)] = str(res)[0]

        if self.step == self.oplen + 1:
            res, _ = self.spec.operation.step(self.op1[-self.step], self.op2[-self.step])
            self.curr_bb_state[4][-(self.step)] = str(res)[0]

        return retval

    def _int_to_list(self, num: int):
        return [int(digit) for digit in str(num)]

    def _pack(self):
        x_pad_start = self.random_x_start * " "
        x_pad_end = (self.spec.width - self.random_x_start - self.oplen - 2) * " "

        ypadstr = self.spec.width * " "
        y_pad_start = self.random_y_start * [ypadstr]
        y_pad_end = (self.spec.height - self.random_y_start - 5) * [ypadstr]

        """Pack the current state of the blackboard into a list of strings."""
        return y_pad_start + [x_pad_start + "".join(x) + x_pad_end for x in self.curr_bb_state] + y_pad_end



class BasicOpBlackboardDataset(GeneratedDataset):
    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer = None,
        regenerate: bool = False,
        generation_spec: GenerationSpec = BASE_GEN_SPEC,
        blackboard_spec: BlackboardSpec = BASE_BLACKBOARD_SPEC,
    ):
        path = path or TRAIN_PATH_BASE.format(blackboard_spec.operation.get_name())
        super().__init__(
            path,
            tokenizer,
            regenerate,
            generation_spec,
        )
        self.specs = blackboard_spec

    @override
    def __generate__(self, spec: GenerationSpec):
        inputs = []
        labels = []

        for _ in range(spec.size):
            a = torch.randint(spec.low, spec.high, (1,)).item()
            b = torch.randint(spec.low, spec.high, (1,)).item()

            input_states, output_states = self._generate_blackboard_pairs(a, b)
            inputs += input_states
            labels += output_states

        return inputs, labels

    def _generate_blackboard_pairs(self, a: Number, b: Number):

        # perform hand addition or subtraction and store the digit steps in lists
        max_input_length = max(np.log10(a).floor(), np.log10(b).floor())

        digits_a = np.empty(max_input_length, dtype=int)
        digits_b = np.empty(max_input_length, dtype=int)
        for i in range(max_input_length):
            digits_a[max_input_length - i - 1] = a % 10
            a //= 10
            digits_b[max_input_length - i - 1] = b % 10
            b //= 10

        blackboards = [*BasicOpBlackboardIterator(digits_a, digits_b, self.specs)]

        return blackboards[:-1], blackboards[1:]
