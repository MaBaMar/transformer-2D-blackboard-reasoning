# ------------------------------------------------------------
# collators.py
#
# Hold collator methods for datasets and model training
#
# I tried to implement it as efficiently as I could.
# ------------------------------------------------------------

import torch
from typing import TypeAlias, Callable, TypeVar, Any

# generics to be compatible with torch's collation function type
_T = TypeVar("_T")  # data sample type
_R = TypeVar("_R")  # collator return type

# type aliases for readability
BBDataSampleType: TypeAlias = dict[str, torch.Tensor]
BBCollatedSampleType: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# ------------------------------------------------------------
# Functional transformations
# ------------------------------------------------------------
def make_collator_with_args(
    func: Callable[..., _R], *args: Any, **kwargs: Any
) -> Callable[[list[_T]], _R]:
    """
    Converts a function with additional arguments into a valid collator function.

    Args:
        func: The function to wrap (e.g., collate_blackboards).
                It must accept a list as its first argument.
        *args: Positional arguments to pass to func after the batch.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        A function that takes a batch (list[_T]) and returns the result of func (_R).
    """

    def collate_fn(batch: list[_T]):
        return func(batch, *args, **kwargs)

    return collate_fn


# ------------------------------------------------------------
# Higher-order collators
# ------------------------------------------------------------
def collate_blackboards(batch: list[tuple[BBDataSampleType, BBDataSampleType]], pad_token_id: int, device: torch.device) -> tuple[BBCollatedSampleType, BBCollatedSampleType]:
    """
    Collate a batch of blackboard data into a tuple of input and output tensors.

    Args:
        batch (list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]): A list of tuples containing input and output data,
            where each data sample consists of a dictionary of tensors with the following keys and shapes:
                - "tokens": torch.Tensor of shape (H, W)
                - "pos_row": torch.Tensor of shape (H*W,)
                - "pos_col": torch.Tensor of shape (H*W,)
        pad_token_id (int): The ID of the padding token, needed for mask generation.
        device (torch.device): The device on which the tensors should be placed.

    Returns:
        tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
            (X,Y) where X is the sample and Y the target state. Moreover, X and Y are tuples of input tensors (tokens, pos_row, pos_col, mask).
    """
    assert len(batch) > 0, "Batch must not be empty"
    assert batch[0][0]["tokens"].shape == batch[0][1]["tokens"].shape, f"Input and output tokens must have the same shape, got {batch[0][0]['tokens'].shape} and {batch[0][1]['tokens'].shape}"

    B, (H,W) = len(batch), batch[0][0]["tokens"].shape
    L = H * W       # length of flattened blackboard

    x_tokensq   = torch.empty((B, L), dtype=torch.long, device=device)
    x_pos_row   = torch.empty((B, L), dtype=torch.long, device=device)
    x_pos_col   = torch.empty((B, L), dtype=torch.long, device=device)
    y_tokensq   = torch.empty((B, L), dtype=torch.long, device=device)
    y_pos_row   = torch.empty((B, L), dtype=torch.long, device=device)
    y_pos_col   = torch.empty((B, L), dtype=torch.long, device=device)

    for i, item in enumerate(batch):
        x_tokensq[i] = item[0]["tokens"].flatten()
        x_pos_row[i] = item[0]["pos_row"]
        x_pos_col[i] = item[0]["pos_col"]
        y_tokensq[i] = item[1]["tokens"].flatten()
        y_pos_row[i] = item[1]["pos_row"]
        y_pos_col[i] = item[1]["pos_col"]

    # despite currently only blackboard sequences in y may contain padding tokens, we still check x for generality and in case we change sth later on
    x_key_padding_mask = x_tokensq == pad_token_id
    y_key_padding_mask = y_tokensq == pad_token_id

    x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (
        x_tokensq,
        x_pos_row,
        x_pos_col,
        x_key_padding_mask
    ) # shapes (B, L), (B, L), (B, L), (B, L)

    y: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (
        y_tokensq,
        y_pos_row,
        y_pos_col,
        y_key_padding_mask
    ) # shapes (B, L), (B, L), (B, L), (B, L)

    return x, y
