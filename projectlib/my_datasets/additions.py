import torch

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import override, TypeAlias, Union, Optional

from projectlib.my_datasets.base import GeneratedDataset, GenerationSpec, Split
from projectlib.my_datasets._operands import OPERATION, Operation
from projectlib.my_datasets.utils import num_to_str



EVAL_PATH = "datasets/additions_eval.pt"



TokenizerType: TypeAlias = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

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
        generation_spec: GenerationSpec,
        path: Optional[str] = None,
        tokenizer: Optional[TokenizerType] = None,
        seed: Optional[int] = None,
        regenerate: bool = True,
        operand: Operation = "+",
        spaces: bool = False,
    ):
        self.operand = operand
        self.spaces = spaces

        path = path if path else EVAL_PATH
        super().__init__(
            path=path,
            tokenizer=tokenizer,
            regenerate=regenerate,
            generation_spec=generation_spec,
            seed=seed,
            disallow_op_permutations=self.operand == "-"
        )

    @override
    def __generate__(self, spec: GenerationSpec, split: Split = Split.EVAL):
        """Generate the addition dataset"""

        if split != Split.EVAL:
            raise ValueError("For this dataset you should only use the evaluation split!")

        inputs = []
        labels = []

        for (a, b) in self.eval_nums:
            func = num_to_str if self.spaces else lambda x: x

            inputs.append(f"{func(a)} {self.operand} {func(b)}")
            labels.append(f"{func(OPERATION[self.operand](a, b))}")

        return inputs, labels
