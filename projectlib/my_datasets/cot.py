import torch

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import override, TypeAlias, Union, Optional

from projectlib.my_datasets.base import GeneratedDataset, GenerationSpec, Split, PaddingMode
from projectlib.my_datasets._operands import OPERATION, Operation
from projectlib.my_datasets.utils import num_to_str, get_digits, digits_to_str



EVAL_PATH = "datasets/scratchpads_eval.pt"
TEST_PATH = "datasets/scratchpads_test.pt"
TRAIN_PATH = "datasets/scratchpads_train.pt"


TokenizerType: TypeAlias = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

class CoTDataset(GeneratedDataset):
    """
    Dataset containing prompts that induce a chain of thought approach that is similar to scratchpads.

    Parameters:
        path (str, optional): Path to store/load the dataset. Defaults to train or eval path.
        tokenizer (AutoTokenizer, optional): Tokenizer for encoding examples. Defaults to None.
        train (bool, optional): Generate training set if True, else evaluation. Defaults to True.
        regenerate (bool, optional): Regenerate dataset if True, even if already present. Defaults to False.
        generation_spec (GenerationSpec, optional): Controls size and numeric range. Defaults to BASE_SPEC.
        operand (Operation, optional): Arithmetic operation ("+" or "-"). Defaults to "+".

    Returns:
        A `ScratchpadDataset` object containing lists of inputs and labels.

    Example:
        The entries are of the following form

        {
            'input': '1 7 + 8 3',

            'label': 'Input: 1 7 + 8 3

                      Target:

                      &lt;scratch&gt;

                      1 7 + 8 3 , C: 0

                      1 + 8 , 0 C: 1 # added 7 + 3 = 0 carry 1

                      , 0 0 C: 1 # added 1 + 8 + 1 = 0 carry 1

                      1 0 0

                      &lt;/scratch&gt;

                      Result: 1 0 0'
        }
    """
    def __init__(
        self,
        generation_spec: GenerationSpec,
        path: Optional[str] = None,
        tokenizer: Optional[TokenizerType] = None,
        split: Split = Split.EVAL,
        seed: Optional[int] = None,
        regenerate: bool = True,
        operand: Operation = "+",
        tokenizer_padding_mode: PaddingMode = "do_not_pad"
    ):
        self.operand = operand

        base_path = None
        match split:
            case Split.EVAL:
                base_path = EVAL_PATH
            case Split.TEST:
                base_path = TEST_PATH
            case Split.TRAIN:
                base_path = TRAIN_PATH

        path = path if path else base_path

        super().__init__(
            path=path,
            tokenizer=tokenizer,
            split=split,
            regenerate=regenerate,
            generation_spec=generation_spec,
            seed=seed,
            disallow_op_permutations=self.operand == "-",
            tokenizer_padding_mode=tokenizer_padding_mode
        )

    @override
    def __generate__(self, spec: GenerationSpec, split: Split = Split.EVAL):
        """Generate the scratchpad dataset"""

        inputs = []
        labels = []
        scratchpads = []

        numbers = []
        match split:
            case Split.EVAL: numbers = self.eval_nums
            case Split.TEST: numbers = self.test_nums
            case Split.TRAIN: numbers = self.train_nums

        for a, b in numbers:
            inputs.append(f"Input: {num_to_str(a)} {self.operand} {num_to_str(b)} <sep>")
            labels.append(OPERATION[self.operand](a, b))
            scratchpads.append(self._generate_scratchpad(a, b))

        if split == Split.EVAL:
            return inputs, labels
        else:
            return scratchpads, labels


    def _generate_scratchpad(self, a: int, b: int) -> str:
        """Generate the scratchpad for a and b"""

        d_a = get_digits(a)
        d_b = get_digits(b)

        n = max(len(d_a), len(d_b))

        d_a = get_digits(a, n)
        d_b = get_digits(b, n)

        # Generate scratchpad line by line
        scratchpad = f""

        result = []
        prev_carry = 0

        for i in range(1, n + 1):
            line, prev_carry = self._generate_line(
                prev_carry=prev_carry,
                result=result,
                curr_a=d_a[n - i],
                curr_b=d_b[n - i],
            )
            scratchpad += line

        if self.operand == "+":
            scratchpad += f"{prev_carry} {digits_to_str(result)}\n"
        elif self.operand == "-":
            scratchpad += f"{digits_to_str(result)}\n"
        else:
            raise NotImplementedError()

        return (
            f"Input: {num_to_str(a)} {self.operand} {num_to_str(b)} <sep>\n"
            f"Computation:\n{scratchpad}\n"
            f"Result: {num_to_str(int(OPERATION[self.operand](a, b)))}\n<eos>"
        )


    def _generate_line(self, prev_carry: int, result: list[int], curr_a: int, curr_b: int) -> tuple[str, int]:
        """Generate the next line of the scratchpad"""

        # Compute result of current two digits
        if self.operand == "+":
            curr_digits = get_digits(curr_a + curr_b + prev_carry)
            operation = f"{curr_a} + {curr_b} + {prev_carry}" if prev_carry else f"{curr_a} + {curr_b}"
        elif self.operand == "-":
            borrow = curr_a < curr_b + prev_carry
            curr_digits = [1] + get_digits((10 + curr_a) - (curr_b + prev_carry)) if borrow else get_digits(curr_a - (curr_b + prev_carry))
            operation = f"1{curr_a} - ({curr_b} + {prev_carry})" if borrow else f"{curr_a} - ({curr_b} + {prev_carry})"
        else:
            NotImplementedError()

        # Add current digit to result
        curr_digit = curr_digits[-1]
        result.insert(0, curr_digit)

        # Compute the carry
        carry = 1 if len(curr_digits) > 1 else 0

        # Generate the line
        line = f"{operation}, {digits_to_str(result)} carry: {carry}\n"

        return line, carry
