import torch

from transformers import AutoTokenizer
from typing import override

from projectlib.my_datasets.base import GeneratedDataset, GenerationSpec
from projectlib.my_datasets._operands import OPERATION, Operation
from projectlib.my_datasets.utils import num_to_str, get_digits, get_number, digits_to_str



EVAL_PATH = "datasets/scratchpads_eval.pt"
TRAIN_PATH = "datasets/scratchpads_train.pt"


BASE_SPEC = GenerationSpec(10, 10, 100)



class ScratchpadDataset(GeneratedDataset):
    """
    Dataset containing prompts that induce a chain of thought approach that is based on scratchpads. 

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
            'input': 'Example:

                      Input: 8 9 + 9 9
            
                      Target:

                      &lt;scratch&gt;

                      8 9 + 9 9 , C: 0

                      8 + 9 , 8 C: 1 # added 9 + 9 = 8 carry 1

                      , 8 8 C: 1 # added 8 + 9 + 1 = 8 carry 1

                      1 8 8

                      &lt;/scratch&gt;

                      Result: 1 8 8

                      
                      Compute: 1 7 + 8 3', 

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
        path: str = None,
        tokenizer: AutoTokenizer = None,
        train: bool = True,
        regenerate: bool = False,
        generation_spec: GenerationSpec = BASE_SPEC,
        operand: Operation = "+",
    ):
        self.operand = operand
        
        path = path if path else (TRAIN_PATH if train else EVAL_PATH)
        super().__init__(
            path=path,
            tokenizer=tokenizer,
            regenerate=regenerate,
            generation_spec=generation_spec,
        )

    @override
    def __generate__(self, spec: GenerationSpec):
        """Generate the scratchpad dataset"""
        
        inputs = []
        labels = []

        numbers = []
        for _ in range(spec.size):
            a = torch.randint(spec.low, spec.high, (1,)).item()
            b = torch.randint(spec.low, spec.high, (1,)).item()

            if(self.operand == "-") and a < b:
                a, b = b, a

            numbers.append((a, b))

        for a, b in numbers:
            c = torch.randint(spec.low, spec.high, (1,)).item()
            d = torch.randint(spec.low, spec.high, (1,)).item()

            if(self.operand == "-") and c < d:
                c, d = d, c

            example_scratchpad = self._generate_scratchpad(c, d)
            target_scratchpad = self._generate_scratchpad(a, b)

            inputs.append((
                f"Example:\n{example_scratchpad}\n"
                f"Compute: {num_to_str(a)} {self.operand} {num_to_str(b)}"
            ))
            labels.append(target_scratchpad)

        return inputs, labels
    
    def _generate_scratchpad(self, a: int, b: int) -> str:
        """Generate the scratchpad for a and b"""

        d_a = get_digits(a)
        d_b = get_digits(b)

        assert len(d_a) == len(d_b), "Input numbers do not have the same amount of digits!"
        n = len(d_a)

        # Generate scratchpad line by line
        scratchpad = f"{num_to_str(a)} {self.operand} {num_to_str(b)} , C: 0\n"

        result = []
        prev_carry = 0

        for i in range(1, n + 1):
            line, prev_carry = self._generate_line(
                prev_carry=prev_carry, 
                result=result, 
                d_a=d_a[:n - i + 1], 
                d_b=d_b[:n - i + 1],
            )
            scratchpad += line

        if self.operand == "+":
            scratchpad += f"{prev_carry} {digits_to_str(result)}\n"
        elif self.operand == "-":
            scratchpad += f"{digits_to_str(result)}\n"
        else:
            raise NotImplementedError()

        return (
            f"Input: {num_to_str(a)} {self.operand} {num_to_str(b)}\n"
            f"Target:\n<scratch>\n{scratchpad}</scratch>\n"
            f"Result: {num_to_str(OPERATION[self.operand](a, b))}\n"
        )
    
    def _generate_line(self, prev_carry: int, result: list[int], d_a: list[int], d_b: list[int]) -> tuple[str, int]:
        """Generate the next line of the scratchpad"""
        
        left_a = digits_to_str(d_a[:-1])
        left_b = digits_to_str(d_b[:-1])
        curr_a = d_a[-1]
        curr_b = d_b[-1]

        # Compute result of current two digits
        if self.operand == "+":
            curr_digits = get_digits(curr_a + curr_b + prev_carry)
        elif self.operand == "-":
            borrow = curr_a < curr_b + prev_carry
            curr_digits = [1] + get_digits((10 + curr_a) - (curr_b + prev_carry)) if borrow else get_digits(curr_a - (curr_b + prev_carry))
        else:
            NotImplementedError()

        # Add current digit to result
        curr_digit = curr_digits[-1]
        result.insert(0, curr_digit)

        # Compute the carry
        carry = 1 if len(curr_digits) > 1 else 0

        # Generate the operation
        operation = f"{left_a} {self.operand} {left_b} " if left_a and left_b else ""
        
        # Generate the comment
        if self.operand == "+":
            if prev_carry:
                comment = f"# added {curr_a} + {curr_b} + 1 = {curr_digit} carry {carry}"
            else:   
                comment = f"# added {curr_a} + {curr_b} = {curr_digit} carry {carry}"
        elif self.operand == "-":
            borrowed_a = (10 + curr_a) if borrow else curr_a
            if prev_carry:
                comment = f"# subtracted {curr_b} + {prev_carry} + {curr_digit} = {borrowed_a} carry {carry}"
            else:   
                comment = f"# subtracted {curr_b} + {curr_digit} = {borrowed_a} carry {carry}"
        else:
            raise NotImplementedError()

        # Generate the line
        line = f"{operation}, {digits_to_str(result)} C: {carry} {comment}\n"
            
        return line, carry
