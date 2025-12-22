"""
Training auxiliaries for the research project. Feel free to define new functions here and to use them in the code.
Please regularly push updated versions of the library, so others can use the same functionality.
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def get_cosine_decay_scheduler_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int):
    """
    linear warmup and then cosine decay scheduler
    """

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=num_warmup_steps)
    decay = CosineAnnealingLR(optimizer, T_max=num_training_steps-num_warmup_steps, eta_min=0.1)

    return SequentialLR(optimizer, [warmup, decay], [num_warmup_steps])

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels[0]
    preds = logits.argmax(dim=-1)

    if labels.shape != preds.shape:
        return 0

    output_wise = (preds == labels).all(dim=1).float().mean().item()

    return output_wise


def compute_accuracy_pt(logits: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels[0]
    preds = logits.argmax(dim=-1)

    if labels.shape != preds.shape:
        return 0

    labels = labels.reshape(-1)
    preds = preds.reshape(-1)

    token_wise = (preds == labels).float().mean().item()

    return token_wise
