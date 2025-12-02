# ------------------------------------------------------------
# bb_chain_wrapper.py
#
# Generate a chain of blackboard reasoning steps. The wrapper function exposes a complete utility
# interface for model prompting to the user, facilitating benchmarking, testing, and debugging.
# ------------------------------------------------------------

from logging import getLogger
import torch
import numpy as np

from projectlib.my_datasets._blackboard_operands import Subtraction
from projectlib.my_datasets.blackboards import BasicOpBlackboardIterator, BlackboardSpec, BBVocabTokenizer, bb_prettyprint
from projectlib.wrappertypes import BBChainGenerator, BBEndOfChainException

ASCII_NUMBERS = "0123456789"

class BBChain:
    tokenizer: BBVocabTokenizer
    steps: list[torch.Tensor]
    board_height: int
    board_width: int


    def __init__(self, tokenizer: BBVocabTokenizer, steps: list[torch.Tensor], board_height: int, board_width: int):
        self.tokenizer = tokenizer
        self.steps = steps
        self.board_height = board_height
        self.board_width = board_width

        self._negres: bool = False       # might need to change the result sign for subtraction
        self._intres_cache: int | None = None
        self._intres_cache_set: bool = False

    def set_negres(self, negres: bool):
        """Sets the negres flag to the given value. """
        self._negres = negres

    @property
    def result(self) -> int | None:
        """
        Returns the result of the BBChain computation as an integer. The result is encoded in the last step of the chain.
        Returns None if the chain is incomplete or the result invalid.
        """

        if(self._intres_cache_set):
            return self._intres_cache

        self._intres_cache_set = True

        if len(self.steps) == 0:
            return None

        # detokenize and extract the result as an integer
        final_state = self.tokenizer.decode(self.steps[-1].view(self.board_height, self.board_width))

        # look for last non-empty line, this line should hold the result
        for i in range(self.board_height-1, 3, -1): # result cannot appear in the first four lines
            if not self._is_empty_line(final_state[i]):
                ans_str: str = "".join(filter(lambda x: x in ASCII_NUMBERS, final_state[i]))
                if(len(ans_str) == 0):
                    return None

                if self._negres:
                    self._intres_cache = -int(ans_str)
                else:
                    self._intres_cache = int(ans_str)
                return self._intres_cache

        # result not found
        return None

    def _is_empty_line(self, line: list[str]) -> bool:
        for token in line:
            if token != self.tokenizer.empty_id:
                return False
        return True

    def show_steps(self):
        print(f"BBChain(board_height={self.board_height}, board_width={self.board_width},\nsteps=[")
        for step in self.steps:
            bb_prettyprint(step.view(self.board_height, self.board_width), self.tokenizer)
        print("])")

class BBChainReasoner:
    def __init__(self, bb_model: BBChainGenerator | str, device: torch.device, bb_spec: BlackboardSpec, timeout_iters: int = 100, tokenizer: BBVocabTokenizer | None = None):
        """End to end algorithmic reasoning wrapper

        Args:
            bb_model (torch.nn.Module | str): The blackboard model to use, either as a pretrained model or a path to a saved model. Should also support the BBChainGenerator interface.
            device (torch.device): The device on which to compute inference.
            bb_spec (BlackboardSpec): The blackboard specification to use for blackboard generation. This MUST be compatible with the blackboard model.
            timeout_iters (int): The maximum number of iterations in any generation process before aborting.
            tokenizer (BBVocabTokenizer | None): The tokenizer to use for blackboard generation. If None, a default tokenizer will be used.
        """
        if isinstance(bb_model, str):
            self.model = torch.load(bb_model).to(device)
        else:
            self.model = bb_model.to(device)

        self.model.eval()
        self.device = device
        self.spec = bb_spec
        self._timeout = timeout_iters
        self._tok = tokenizer or BBVocabTokenizer()
        self.logger = getLogger(__name__)

    def compute_from_blackboard(self, blackboard: torch.Tensor) -> BBChain:
        """
        Computes a BBChain from a blackboard.

        Args:
            blackboard (torch.Tensor): The blackboard to compute from. Sould be either of shape (H,W) or (H*W).

        Returns:
            BBChain: The computed BBChain.
        """

        assert blackboard.shape == (self.spec.height, self.spec.width) or blackboard.shape == (self.spec.height * self.spec.width,), "Blackboard shape must match specification (you may need to remove the batch dimension)"

        blackboard = blackboard.flatten().to(self.device)
        H, W = self.spec.height, self.spec.width
        _pos_row = torch.arange(H*W, device=self.device, dtype=torch.long)
        _pow_col = torch.arange(H*W, device=self.device, dtype=torch.long).view(H, W).T.flatten()
        _pad_mask = blackboard == self._tok.pad_id

        # add a batch dimension of 1 and do inference
        return self.compute_from_input((blackboard[None, ...], _pos_row[None, ...], _pow_col[None, ...], _pad_mask[None, ...]))

    def compute_from_operands(self, operand1: int, operand2: int) -> BBChain:

        assert operand1 >= 0 and operand2 >= 0, "Operands must be non-negative"

        swapped: bool = False

        # order the operands
        if isinstance(self.spec.operation, Subtraction) and operand1 < operand2:
            operand1, operand2 = operand2, operand1
            swapped = True

        # convert operands to blackboard
        max_input_length = max(np.floor(np.log10(operand1)).astype(np.int32), np.floor(np.log10(operand2)).astype(np.int32)) + 1

        # fast conversion
        digits_a = np.empty(max_input_length, dtype=int)
        digits_b = np.empty(max_input_length, dtype=int)
        for i in range(max_input_length):
            digits_a[max_input_length - i - 1] = operand1 % 10
            operand1 //= 10
            digits_b[max_input_length - i - 1] = operand2 % 10
            operand2 //= 10

        opframe_generator = BasicOpBlackboardIterator(digits_a, digits_b, self.spec.operation)
        # only care about first state
        st: list[list[str]] = next(opframe_generator)

        # randomize the position
        p_x = 0
        p_y = 0

        if self.spec.randomize_position:
            r_max_x = self.spec.width - opframe_generator.frame_width
            r_max_y = self.spec.height - opframe_generator.frame_height
            if r_max_x > 0:
                p_x = np.random.randint(0, r_max_x)
            if r_max_y > 0:
                p_y = np.random.randint(0, r_max_y)

        blackboard = torch.full((self.spec.height, self.spec.width), self._tok.empty_id, dtype=torch.long, device=self.device)
        blackboard[p_y:p_y + opframe_generator.frame_height, p_x:p_x + opframe_generator.frame_width] = self._tok.encode(st)

        # set BOS token
        blackboard[0, 0] = self._tok.bos_id

        res: BBChain = self.compute_from_blackboard(blackboard)
        res.set_negres(swapped)

        return res

    def compute_from_input(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> BBChain:
        """
        Compute a blackboard chain from a given input.

        Args:
            x: A tuple of tensors representing the input. Expected to be on the right device.

        Returns:
            A BBChain object.
        """
        steps: list[torch.Tensor] = []

        for i in range(self._timeout):
            steps.append(x[0])
            self.logger.debug(f"Step {i}: attempting to generate next state")               # some logging. Probably helpful for later
            try:    # try to generate the next state
                bb_next = self.model.next_state(x)
                self.logger.debug(f"generated state\n{bb_next}")
                x = (bb_next, *x[1:])
                self.logger.debug(f"Step {i}: next state successfully generated")
            except BBEndOfChainException:   # next state is END OF BOARD CHAIN
                # signal indicates that an End Of Chain state has been reached! we want to get here!
                self.logger.debug(f"Step {i}: reached end of chain")
                return BBChain(self._tok, steps, self.spec.height, self.spec.width)

        self.logger.warning("Reasoning chain generation timed-out after {} iterations. Please make sure your model is properly trained.".format(self._timeout))
        return BBChain(self._tok, steps, self.spec.height, self.spec.width)
