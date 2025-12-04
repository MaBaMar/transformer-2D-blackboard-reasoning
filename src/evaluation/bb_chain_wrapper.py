# ------------------------------------------------------------
# bb_chain_wrapper.py
#
# Generate a chain of blackboard reasoning steps. The wrapper function exposes a complete utility
# interface for model prompting to the user, facilitating benchmarking, testing, and debugging.
#
# ------------------------------------------------------------

from logging import getLogger
import torch
import numpy as np

from projectlib.my_datasets._blackboard_operands import Subtraction
from projectlib.my_datasets.blackboards import BasicOpBlackboardIterator, BlackboardSpec, BBVocabTokenizer, bb_prettyprint, operands_to_bbchaingen
from projectlib.wrappertypes import BBChainGenerator

ASCII_NUMBERS = "0123456789"

class BBChain:
    tokenizer: BBVocabTokenizer
    board_height: int
    board_width: int

    def __init__(self, tokenizer: BBVocabTokenizer, board_height: int, board_width: int, steps: list[torch.Tensor] | None):
        # public members
        self.tokenizer = tokenizer
        self.board_height = board_height
        self.board_width = board_width

        # logically private members
        self._steps: list[torch.Tensor] = steps or []
        self._negres: bool = False              # might need to change the result sign for subtraction
        self._intres_cache: float = torch.nan   # would be int, but float supports NaN, which is useful for handling errors
        self._intres_cache_set: bool = False

    def add_step(self, step: torch.Tensor):
        self._steps.append(step)

    def set_negres(self, negres: bool):
        """Sets the negres flag to the given value. """
        self._negres = negres

    @property
    def result(self) -> float:
        """
        Returns the result of the BBChain computation as an integer. The result is encoded in the last step of the chain.
        Returns None if the chain is incomplete or the result invalid.
        """

        if(self._intres_cache_set):
            return self._intres_cache

        self._intres_cache_set = True

        if len(self._steps) == 0:
            return torch.nan

        # detokenize and extract the result as an integer
        final_state = self.tokenizer.decode(self._steps[-1].view(self.board_height, self.board_width))

        # look for last non-empty line, this line should hold the result
        for i in range(self.board_height-1, 3, -1): # result cannot appear in the first four lines
            if not self._is_empty_line(final_state[i]):
                ans_str: str = "".join(filter(lambda x: x in ASCII_NUMBERS, final_state[i]))
                if(len(ans_str) == 0):
                    return torch.nan

                if self._negres:
                    self._intres_cache = -int(ans_str)
                else:
                    self._intres_cache = int(ans_str)
                return self._intres_cache

        # result not found
        return torch.nan

    def _is_empty_line(self, line: list[str]) -> bool:
        for token in line:
            if token != self.tokenizer.empty_id:
                return False
        return True

    def show_steps(self):
        print(f"BBChain(board_height={self.board_height}, board_width={self.board_width},\nsteps=[")
        for step in self._steps:
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

    def compute_from_single_blackboard(self, blackboard: torch.Tensor) -> BBChain:
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
        return self.compute_from_databatch((blackboard[None, ...], _pos_row[None, ...], _pow_col[None, ...], _pad_mask[None, ...]))[0]

    def compute_from_operands(self, operand1: int, operand2: int) -> BBChain:

        assert operand1 >= 0 and operand2 >= 0, "Operands must be non-negative"

        swapped: bool = False

        # order the operands
        if isinstance(self.spec.operation, Subtraction) and operand1 < operand2:
            operand1, operand2 = operand2, operand1
            swapped = True

        gen, p_x, p_y = operands_to_bbchaingen(operand1, operand2, self.spec)

        blackboard = torch.full((self.spec.height, self.spec.width), self._tok.empty_id, dtype=torch.long)
        blackboard[p_y:p_y + gen.frame_height, p_x:p_x + gen.frame_width] = self._tok.encode(next(gen))

        res: BBChain = self.compute_from_single_blackboard(blackboard)
        res.set_negres(swapped)

        return res

    def compute_from_databatch(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> list[BBChain]:
        """
        Compute a blackboard chain for each of the given inputs.

        Args:
            x: A tuple of tensors representing the input. Expected to be on the right device. Corresponds to the batched input of the blackboard dataset.

        Returns:
            A list of BBChain objects.
        """
        B, _ = x[0].shape
        chains: list[BBChain] = [BBChain(self._tok, self.spec.height, self.spec.width, [x[0][i]]) for i in range(B)]
        propagation_indices = torch.arange(B, dtype=torch.long, device=self.device)

        for stidx in range(self._timeout):
            self.logger.debug(f"Step {stidx}: generate next state")               # some logging. Probably helpful for later
            bb_next = self.model.next_state(x) # dimension [B,L]

            # only propagate the state if it is not an end of chain state
            state_propagation_mask = ~(bb_next[:, 0] == self._tok.eos_id).flatten()

            # keep track of the chains that require further propagation
            propagation_indices = propagation_indices[state_propagation_mask]

            x = (
                bb_next[state_propagation_mask, :],
                x[1][state_propagation_mask, :],
                x[2][state_propagation_mask, :],
                x[3][state_propagation_mask, :]
            )

            # add steps to chains
            for i in range(x[0].shape[0]):
                chains[propagation_indices[i]].add_step(bb_next[i].clone())

            if not state_propagation_mask.any():
                # all chains terminated
                self.logger.debug(f"successfully completed all chains")
                return chains

        self.logger.warning(f"Reasoning chain generation timed-out after {self._timeout} iterations. {x[0].shape[0]} chains remain incomplete. They will appear truncated. Please make sure your model is properly trained.")
        return chains


# ------------------------------------------------------------
# handy utility functions
# ------------------------------------------------------------

def chainlist_to_results(chainlist: list[BBChain]) -> torch.Tensor:
    """Convert a list of BBChain objects with length B to a torch tensor of shape [B, 1] containing the extracted result for each chain.

    Args:
        chainlist: A list of BBChain objects.

    Returns:
        A list of dictionaries, where each dictionary represents a chain.
    """
    return torch.tensor([chain.result for chain in chainlist])[None, ...]
