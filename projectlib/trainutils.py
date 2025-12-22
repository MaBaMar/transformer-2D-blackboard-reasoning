"""
Training auxiliaries for the research project. Feel free to define new functions here and to use them in the code.
Please regularly push updated versions of the library, so others can use the same functionality.
"""

import torch
import math
import warnings
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_decay_scheduler_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_tokens: int | None = None) -> float:
    preds = logits.argmax(dim=-1)

    if labels.shape != preds.shape:
        warnings.warn("Mismatched shapes between logits and labels") # good for debugging purposes
        return 0

    base = (preds == labels)
    if ignore_tokens is not None:
        base |= (labels == ignore_tokens) # set all ignored tokens to True (logically equivalent to ignoring them)
    output_wise = base.all(dim=1).float().mean().item()

    return output_wise


def compute_accuracy_pt(logits: torch.Tensor, labels: torch.Tensor, ignore_tokens: int | None = None) -> float:
    preds = logits.argmax(dim=-1)

    if labels.shape != preds.shape:
        warnings.warn("Mismatched shapes between logits and labels") # good for debugging purposes
        return 0

    labels = labels.reshape(-1)
    preds = preds.reshape(-1)

    base = (preds == labels)
    if ignore_tokens is not None:
        base = base[labels != ignore_tokens]
    token_wise = base.float().mean().item()

    return token_wise
