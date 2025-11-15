import torch

from transformers import AutoTokenizer
from typing import override

from projectlib.my_datasets.base import GeneratedDataset, GenerationSpec
from projectlib.my_datasets._operands import OPERATION, Operation
from projectlib.my_datasets.utils import num_to_str



EVAL_PATH = "datasets/additions_eval.pt"
TRAIN_PATH = "datasets/additions_train.pt"


BASE_SPEC = GenerationSpec(10, 10, 20)



class AdditionDataset(GeneratedDataset):
    """
    Dataset containing prompts for basic operations (addition/subtraction).

    Parameters:
        path (str, optional): Path to store/load the dataset. Defaults to train or eval path.
        tokenizer (AutoTokenizer, optional): Tokenizer for encoding examples. Defaults to None.
        train (bool, optional): Generate training set if True, else evaluation. Defaults to True.
        regenerate (bool, optional): Regenerate dataset if True, even if already present. Defaults to False.
        generation_spec (GenerationSpec, optional): Controls size and numeric range. Defaults to BASE_SPEC.
        operand (Operation, optional): Arithmetic operation ("+" or "-"). Defaults to "+".
        spaces (bool, optional): Use spaces between digits if True. Defaults to False.

    Returns:
        A `GeneratedDataset` object containing lists of inputs and labels.

    Example:
        The entries are of the following form

        {'input': '12 + 13', 'label': '25'}

        {'input': '1 2 + 1 3', 'label': '2 5'}  # if spaces=True
    """

    def __init__(
        self,
        path: str = None,
        tokenizer: AutoTokenizer = None,
        train: bool = True,
        regenerate: bool = False,
        generation_spec: GenerationSpec = BASE_SPEC,
        operand: Operation = "+",
        spaces: bool = False,
    ):
        self.operand = operand
        self.spaces = spaces
        
        path = path if path else (TRAIN_PATH if train else EVAL_PATH)
        super().__init__(
            path=path,
            tokenizer=tokenizer,
            regenerate=regenerate,
            generation_spec=generation_spec,
        )

    @override
    def __generate__(self, spec: GenerationSpec):
        """Generate the addition dataset"""
        
        inputs = []
        labels = []

        for _ in range(spec.size):
            a = torch.randint(spec.low, spec.high, (1,)).item()
            b = torch.randint(spec.low, spec.high, (1,)).item()

            if(self.operand == "-") and a < b:
                a, b = b, a

            func = num_to_str if self.spaces else lambda x: x

            inputs.append(f"{func(a)} {self.operand} {func(b)}")
            labels.append(f"{func(OPERATION[self.operand](a, b))}")

        return inputs, labels
