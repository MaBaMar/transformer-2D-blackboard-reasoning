# ------------------------------------------------------------
# bb_chain_wrapper.py
#
# Generate a chain of blackboard reasoning steps. The wrapper function exposes a complete utility
# interface for model prompting to the user, facilitating benchmarking, testing, and debugging.
# ------------------------------------------------------------

from dataclasses import dataclass
import torch

from projectlib.my_datasets.blackboards import BlackboardSpec

@dataclass
class BBChain:
    steps: list[str]

    @property
    def result(self) -> int:
        # detokenize and extract the result as an integer
        # TODO: implement this method
        raise NotImplementedError("BBChain.result is not implemented")

class BBChainReasoner:
    def __init__(self, bb_model: torch.nn.Module | str, device: torch.device, bb_spec: BlackboardSpec):
        """End to end algorithmic reasoning wrapper

        Args:
            bb_model (torch.nn.Module | str): The blackboard model to use, either as a pretrained model or a path to a saved model.
            device (torch.device): The device on which to compute inference.
            bb_spec (BlackboardSpec): The blackboard specification to use for blackboard generation. This MUST be compatible with the blackboard model.
        """
        if isinstance(bb_model, str):
            self.model = torch.load(bb_model).to(device)
        else:
            self.model = bb_model.to(device)

        self.model.eval()
        self.device = device
        self.spec = bb_spec

    def compute_from_blackboard(self, blackboard: torch.Tensor) -> BBChain:
        # TODO: implement this method
        raise NotImplementedError("BBChainReasoner.compute_from_blackboard is not implemented")

    def compute_from_operands(self, operand1: int, operand2: int) -> BBChain:
        # TODO: implement this method
        raise NotImplementedError("BBChainReasoner.compute_from_operands is not implemented")
