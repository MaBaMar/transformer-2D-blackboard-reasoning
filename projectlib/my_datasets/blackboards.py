# ------------------------------------------------------------
# blackboard.py
#
# Generates blackboards to feed to our 2DTPE models. Assumes encoder/decoder architectures as outlined in
# https://github.com/MaBaMar/transformer-2D-blackboard-reasoning/issues/11#issuecomment-3538528978
#
# Terminology:
#   The operation frame is the area of the blackboard that contains the operation symbols and digits.
#   The blackboard frame is the area of the entire blackboard including empty spaces.
#
# Tokenization:
#   Uses a customized tokenization procedure. Currently, we use a simple dummy tokenizer.
#   This will have to be replaced with a more sophisticated tokenizer.
#
# Notes:
#   - We're currently ignoring EOS and BOS tokens. We might need to add them later.
#
# Warning: For use with the dedicated blackboard models only!
# ------------------------------------------------------------

import copy
from dataclasses import dataclass
from typing import Optional
from logging import getLogger, Logger

import numpy as np
import torch
from tqdm import tqdm

from projectlib.my_datasets._blackboard_operands import Addition, CarryOperation, Subtraction
from projectlib.my_datasets.base import GeneratedDataset, GenerationSpec, Split

@dataclass
class BlackboardSpec:
    """Class that specifies the parameters for the blackboard generation."""
    height: int
    width: int
    randomize_position: bool
    operation: CarryOperation

# ------------------------------------------------------------
# configuration
# ------------------------------------------------------------
# blackboard
EVAL_PATH_BASE       = "datasets/bb_{}_eval.pt"
TEST_PATH_BASE       = "datasets/bb_{}_test.pt"
TRAIN_PATH_BASE      = "datasets/bb_{}_train.pt"
BASE_GEN_SPEC        = GenerationSpec(10, 10, 20)
BB_OPFRAME_HEIGHT    = 5    # this is the minimum number of rows required to fit the operation frame
BASE_BLACKBOARD_SPEC = BlackboardSpec(5, 15, False, Addition())

# tokenization, defines vocabulary layout
BB_EMPTY_TOKEN          = "<EMPTY>"    # this token is a placeholder for blackboard grid spaces that are empty, it is different from a mere padding token
BB_BOS_TOKEN            = "<BOS>"      # this token is a placeholder for the beginning of a blackboard sequence
BB_EOS_TOKEN            = "<EOS>"      # this token is a placeholder for the end of a blackboard sequence
BB_PAD_TOKEN            = "<PAD>"      # this token is a placeholder for padding
BB_OPTOKEN_LIST         = ["+", "-"]   # defines supported operator tokens
BB_FILL_NUM_TOKEN       = "_"          # this token indicates placeholders for numbers to potentially be filled in
BB_OPLINE_SEG_TOKEN     = "="          # this token indicates a segment of the computation line separating operands and carry from the result
BB_MODE_ADVANCE_TOKEN   = "<ADV>"      # mode token: advance to next state
BB_MODE_CHECK_TOKEN     = "<CHK>"      # mode token: check/correct current state

# ------------------------------------------------------------
# Simple tokenizer
#
# We implement a simple vocab for manual tokenization on the grid
# We could construct a tokenizer based on the RustTokenizers (PreTrainedTokenizerFast), but this is quite some work and not really necessary)
#
# ------------------------------------------------------------
class BBVocabTokenizer:
    def __init__(self, additional_tokens: list[str] | None = None):
        self.token_to_id: dict[str, int] = {
            BB_BOS_TOKEN: 0,
            BB_EOS_TOKEN: 1,
            BB_EMPTY_TOKEN: 2,
            BB_PAD_TOKEN: 3,
            **{str(i): i + 4 for i in range(10)},
            **{operand_token: i + 14 for i, operand_token in enumerate(BB_OPTOKEN_LIST)},
            BB_FILL_NUM_TOKEN: len(BB_OPTOKEN_LIST) + 14,
            BB_OPLINE_SEG_TOKEN: len(BB_OPTOKEN_LIST) + 15,
            BB_MODE_ADVANCE_TOKEN: len(BB_OPTOKEN_LIST) + 16,
            BB_MODE_CHECK_TOKEN: len(BB_OPTOKEN_LIST) + 17,
            **{token: len(BB_OPTOKEN_LIST) + 18 + i for i, token in enumerate(additional_tokens or [])},
        }
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}

    def encode(self, bb_grid: list[list[str]]) -> torch.Tensor:
        """
        Encode a grid of strings into a tensor of token IDs.

        Args:
            bb_grid (List[List[str]]): A grid of strings, where each string represents a token.

        Returns:
            torch.Tensor: A tensor of token IDs. Will be of type torch.long.

        Raises:
            ValueError: If the input grid contains an unknown token.
        """
        H, W = len(bb_grid), len(bb_grid[0])
        tokens = torch.empty((H, W), dtype=torch.long)
        try:
            for i in range(H):
                for j in range(W):
                    tokens[i, j] = self.token_to_id[bb_grid[i][j]]
        except KeyError as e:
            raise ValueError(f"Unknown token '{e.args[0]}' in grid") from e
        return tokens

    def decode(self, tokens: torch.Tensor) -> list[list[str]]:
        """
        Decode a tensor of token IDs into a grid of strings.

        Args:
            tokens (torch.Tensor): A tensor of token IDs. Should be of type torch.long.

        Returns:
            List[List[str]]: A grid of strings, where each string represents a token.

        Raises:
            ValueError: If the input tensor contains an unknown token ID.
        """
        assert tokens.dtype == torch.long, f"Expected torch.long tensor, got {tokens.dtype}"
        H, W = tokens.shape
        bb_grid = [[''] * W for _ in range(H)]
        try:
            for i in range(H):
                for j in range(W):
                    bb_grid[i][j] = self.id_to_token[int(tokens[i, j].item())]
        except KeyError as e:
            raise ValueError(f"Unknown token ID '{e.args[0]}' in tensor") from e
        return bb_grid

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @property
    def bos_id(self):
        return self.token_to_id[BB_BOS_TOKEN]

    @property
    def eos_id(self):
        return self.token_to_id[BB_EOS_TOKEN]

    @property
    def pad_id(self):
        return self.token_to_id[BB_PAD_TOKEN]

    @property
    def empty_id(self):
        return self.token_to_id[BB_EMPTY_TOKEN]

# ------------------------------------------------------------
# Blackboard solution-steps-sequence generation
# ------------------------------------------------------------

class BasicOpBlackboardIterator:
    def __init__(
        self,
        operand1: np.ndarray,
        operand2: np.ndarray,
        operation: CarryOperation,
        eos_filler: str = BB_PAD_TOKEN,
    ):
        """
        Iterator that generates a sequence of operation frames encoding the blackboard solution-steps of the specified operations.

        Args:
            operand1 (np.ndarray): The first operand encoded as digit array.
            operand2 (np.ndarray): The second operand encoded as digit array.
            operation (CarryOperation): The operation to be performed.

        Assumptions:
            - The operation arrays must be of equal length. Use zero padding if necessary.
        """

        # some debugging checks
        assert len(operand1) == len(operand2), "Operation arrays must be of equal length. Use zero padding"

        self.frame_width: int = len(operand1) + 2
        self.frame_height: int = BB_OPFRAME_HEIGHT

        self.step: int = 0
        self.last_carry: int = 0
        self.operand1: np.ndarray = operand1
        self.operand2: np.ndarray = operand2
        self.oplen: int = len(operand1)
        self.operation: CarryOperation = operation

        # generate the starting operation frame, we will gradually fill in digits while generating the states
        self._reset_operation_frame()

        # this is the EOS state
        self.eos_frame: list[list[str]] = [[eos_filler for _ in range(self.frame_width)] for _ in range(self.frame_height)]
        self.eos_frame[0][0] = BB_EOS_TOKEN

    def __iter__(self):
        """
        Initialize the iterator by resetting the step counter and resetting the operation frame.
        Warning: We do not copy the iterator, but instead reset ourselves. This is fine for our usage, but be careful when using this iterator in multiple places.
        """
        self.step = 0
        self._reset_operation_frame()
        return self

    def __next__(self):

        # we generate the input and output states, as well as `num_digits` intermediate states
        if self.step == self.oplen + 2:
            raise StopIteration

        # current operation frame is what we output
        curr_state = copy.deepcopy(self.curr_operation_frame)   # list is mutable, need to copy
        self.step += 1

        # already generate the next state
        if self.step <= self.oplen:
            res, carry = self.operation.step(self.operand1[-self.step], self.operand2[-self.step], self.last_carry)
            self.curr_operation_frame[2][-(self.step+1)] = str(carry)[0]
            self.curr_operation_frame[4][-(self.step)] = str(res)[0]
            self.last_carry = carry

        if self.step == self.oplen + 1:
            self.curr_operation_frame[4][-(self.step)] = self.operation.finalize_carry(self.last_carry)

        return curr_state

    # ---------------- helpers (logically private methods) ----------------
    def _reset_operation_frame(self) -> None:
        """Reset the operation frame to its initial state, i.e. the input blackboard frame."""
        self.curr_operation_frame: list[list[str]] = [
            2*[BB_EMPTY_TOKEN] + [str(x) for x in self.operand1],                       # operand1 line
            [str(self.operation), BB_EMPTY_TOKEN] + [str(x) for x in self.operand2],    # operand2 line
            [BB_EMPTY_TOKEN]+(self.oplen)*[BB_FILL_NUM_TOKEN]+[BB_EMPTY_TOKEN],         # carry line
            (self.oplen+2) * [BB_OPLINE_SEG_TOKEN],                                     # result separator line
            [BB_EMPTY_TOKEN]+[BB_FILL_NUM_TOKEN]*(self.oplen+1)                         # result line
        ]


# ------------------------------------------------------------
# The actual dataset class to use during training and evaluation
#
# IMPORTANT: Unlike the other GeneratedDataset classes, this class performs tokenization BEFORE saving the dataset.
#            This is for efficiency as we use a custom tokenizer (not the HuggingFace fast tokenizer implemented in rust)
# ------------------------------------------------------------
class TokenizedBlackboardDataset(GeneratedDataset):

    def __init__(
        self,
        path: Optional[str] = None,
        seed: Optional[int] = None,
        split: Split = Split.EVAL,
        regenerate: bool = True,
        generation_spec: GenerationSpec = BASE_GEN_SPEC,
        blackboard_spec: BlackboardSpec = BASE_BLACKBOARD_SPEC,
        additional_tokens: list[str] | None = None
    ):
        base_path = None
        match split:
            case Split.EVAL:
                base_path = EVAL_PATH_BASE.format(blackboard_spec.operation.get_name())
            case Split.TEST:
                base_path = TEST_PATH_BASE.format(blackboard_spec.operation.get_name())
            case Split.TRAIN:
                base_path = TRAIN_PATH_BASE.format(blackboard_spec.operation.get_name())

        path = path or base_path

        self.bb_spec: BlackboardSpec = blackboard_spec
        self.bb_2D_tokenizer: BBVocabTokenizer = BBVocabTokenizer(additional_tokens or [])

        # add a logger for some helpful logging that can be disabled
        self._datalogger: Logger = getLogger(__name__)

        # generate data
        super().__init__(
            path=path,
            regenerate=regenerate,
            generation_spec=generation_spec,
            split=split,
            seed=seed,
            disallow_op_permutations=self.bb_spec.operation.get_name() == 'subtraction'
        )

    def __generate__(self, spec: GenerationSpec, split: Split = Split.EVAL) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor] | int]]:

        inputs: list[dict[str, torch.Tensor]] = []
        labels: list[dict[str, torch.Tensor] | int] = []

        input_operands: list[tuple[int, int]] = []
        match split:
            case Split.EVAL: input_operands = self.eval_nums
            case Split.TRAIN: input_operands = self.train_nums
            case Split.TEST: input_operands = self.test_nums

        for a, b in tqdm(input_operands, desc="Generating data"):
            # size is interpreted as the number of blackboard computation chains to generate
            input_states, output_states = self._generate_blackboard_pairs(a, b, split != Split.EVAL)
            inputs += input_states
            labels += output_states

        # helpful logging to inform the user about what type of data they generated
        if(split == Split.EVAL):
            self._datalogger.info(f"Generated {len(inputs)} blackboard input state - integer pairs")
        else:
            self._datalogger.info(f"Generated {split.size(spec)} blackboard chains encoded in {len(inputs)} data samples")

        return inputs, labels

    def __getitem__(self, idx: int):

        if(idx < 0 or idx >= len(self)):
            raise IndexError("Index out of range")

        return self.data[idx], self.labels[idx]

    # ---------------- helpers (logically private methods) ----------------
    def _generate_blackboard_pairs(self, a: int, b: int, generate_full_chain: bool) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor] | int]]:

        opframe_generator, p_x, p_y = operands_to_bbchaingen(a, b, self.bb_spec)

        # generate solution sequence
        bb_chain: list[torch.Tensor] = []

        for opframe in opframe_generator:

            # extend the blackboard to its full size
            blackboard = torch.full((self.bb_spec.height, self.bb_spec.width), self.bb_2D_tokenizer.empty_id, dtype=torch.long)
            blackboard[p_y:p_y + opframe_generator.frame_height, p_x:p_x + opframe_generator.frame_width] = self.bb_2D_tokenizer.encode(opframe)

            bb_chain.append(blackboard)

            if not generate_full_chain: # we only care about the first blackboard for the evaluation set
                break

        if generate_full_chain:
            # add EOS blackboard
            blackboard = torch.full((self.bb_spec.height, self.bb_spec.width), self.bb_2D_tokenizer.pad_id, dtype=torch.long)
            blackboard[0, 0] = self.bb_2D_tokenizer.eos_id
            bb_chain.append(blackboard)

            # Note: we need copies for the label to avoid aliasing
            return self._pack_to_input_format(bb_chain[:-1]), self._pack_to_input_format(bb_chain[1:], copy=True)
        else:
            return self._pack_to_input_format(bb_chain), [self.bb_spec.operation.on_ints(a, b)]

    def _pack_to_input_format(self, bbs: list[torch.Tensor], copy: bool = False) -> list[dict[str, torch.Tensor]]:

        H, W = bbs[0].shape

        return [{
            "tokens": bb.clone() if copy else bb,
            "pos_row": torch.arange(H*W, dtype=torch.long),
            "pos_col": torch.arange(H*W, dtype=torch.long).view(H, W).T.flatten()
        } for bb in bbs]



# ------------------------------------------------------------
# Blackboard specific utility functions.
#
# They are not in the utils.py file to avoid circular imports.
# ------------------------------------------------------------

def operands_to_bbchaingen(a: int, b: int, bb_spec: BlackboardSpec) -> tuple[BasicOpBlackboardIterator, int, int]:
    """
    Converts two numbers into a blackboard iterator and returns the iterator, as well as the randomized top-left-corner of the blackboard frame.

    Args:
        a (int): The first operand.
        b (int): The second operand.
        bb_spec (BlackboardSpec): The blackboard specification.

    Returns:
        tuple[BasicOpBlackboardIterator, int, int]: [it, x, y] where it is the blackboard iterator corresponding to `a (op) b` where op is specified in `bb_spec` and x, y are the randomized top-left-corner of the blackboard frame.
        If the blackboard specification does not enable randomization, the top-left-corner is set to (0, 0).
    """

    # perform hand addition or subtraction and store the digit steps in lists
    max_input_length: int = max(np.floor(np.log10(a)).astype(np.int32), np.floor(np.log10(b)).astype(np.int32)) + 1

    digits_a = np.empty(max_input_length, dtype=int)
    digits_b = np.empty(max_input_length, dtype=int)
    for i in range(max_input_length):
        digits_a[max_input_length - i - 1] = a % 10
        a //= 10
        digits_b[max_input_length - i - 1] = b % 10
        b //= 10

    opframe_generator = BasicOpBlackboardIterator(digits_a, digits_b, bb_spec.operation)

    if(bb_spec.width < opframe_generator.frame_width or bb_spec.height < opframe_generator.frame_height):
        raise ValueError(f"Generated opframes will not fit the requested dimensions of {bb_spec.height} x {bb_spec.width}. \
                           Input numbers require a size of at least {opframe_generator.frame_height} x {opframe_generator.frame_width}")

    # randomize the position
    p_x = 0
    p_y = 0

    if bb_spec.randomize_position:
        r_max_x = bb_spec.width - opframe_generator.frame_width
        r_max_y = bb_spec.height - opframe_generator.frame_height
        if r_max_x > 0:
            p_x = np.random.randint(0, r_max_x)
        if r_max_y > 0:
            p_y = np.random.randint(0, r_max_y)

    return opframe_generator, p_x, p_y


def bb_prettyprint(board: torch.Tensor, tokenizer: Optional[BBVocabTokenizer] = None):
    """
    Pretty-print a tokenized blackboard

    Args:
        board (torch.Tensor): The board to print.
    """
    tok = tokenizer or BBVocabTokenizer()

    decoded_board = tok.decode(board)
    print((board.shape[1]+2) * "-")
    for i in range(len(decoded_board)):
        line = '|'
        for j in range(len(decoded_board[i])):
            if(decoded_board[i][j] == BB_BOS_TOKEN):
                line += 'B'
            elif(decoded_board[i][j] == BB_EOS_TOKEN):
                line += 'E'
            elif(decoded_board[i][j] == BB_PAD_TOKEN):
                line += 'P'
            elif(decoded_board[i][j] == BB_EMPTY_TOKEN):
                line += ' '
            else:
                line += decoded_board[i][j]
        print(line+'|')
    print((board.shape[1]+2) * "-")


def bb_datasample_prettyprint(sample: dict[str, torch.Tensor]):
    """
    Pretty-prints a dataset sample

    Args:
        sample (Dict[str, torch.Tensor]): The sample to print.
    """

    print("Blackboard state:")
    bb_prettyprint(sample["tokens"])

    print("Blackboard positions:")
    print("pos_row: ", sample["pos_row"])
    print("pos_col: ", sample["pos_col"])
