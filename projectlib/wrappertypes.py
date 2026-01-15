# ------------------------------------------------------------
# wrappertypes.py
#
# Collection of utility types and interfaces for the blackboard reasoning wrapper.
# The wrapper implementations themselves are not part of the library
# and implemented in src/
# ------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import final

import torch

# ------------------------------------------------------------
# Custom Exceptions
# ------------------------------------------------------------

@final
class EndOfComputationException(Exception):
    """Raised when the end of the blackboard chain is reached."""

    def __init__(self, message: str = "End of blackboard chain reached"):
        self.message = message
        super().__init__(self.message)

# ------------------------------------------------------------
# Interfaces for the models -> ensures compatibility with the wrapper functionality
# ------------------------------------------------------------
class BBChainGenerator(ABC, torch.nn.Module):

    @abstractmethod
    def next_state(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Takes a blackboard state and generates the next state in the sequence.

        Args:
            x (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Input data (blackboard, pos_row, pos_col, padding_mask) (current blackboard state).
            The input data is expected to be batched. Specifically, the blackboard tensor should have shape [B, L] where L = H * W.

        Returns:
            torch.Tensor: Next blackboard state.
        """
        raise NotImplementedError()
