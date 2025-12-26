"""
Training auxiliaries for the research project. Feel free to define new functions here and to use them in the code.
Please regularly push updated versions of the library, so others can use the same functionality.
"""

import torch
import warnings

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
