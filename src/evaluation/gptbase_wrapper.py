# ------------------------------------------------------------
# gptbase_wrapper.py
#
# Contains utility functionality for inference passes on the GPTStyleBaseline model.
# Similar format as the bb_chain_wrapper.py file for blackboard reasoning.
# Warning: Inference is very slow and should be used sparingly.
# ------------------------------------------------------------

import torch
from logging import getLogger

from src.models.gptbase import GPTBaseTokenizer, GPTStyleBaseline
from projectlib.utils import ASCII_NUMBERS


class GPTBaseInferenceBatch:
    def __init__(self, computations: list[str]):
        """
        Result batch returned by the GPT wrapper. This class provides a convenient way to store and retrieve the results of a batch of computations.

        Args:
            computations (list[str]): A list of computations to be evaluated. These computations should directly stem from a GPTStyleBaseline model using `batch_inference`
            and tokenizer.strip_decode(...)
        """
        self.computations = computations
        self.result_cache: torch.Tensor | None = None

    @property
    def results(self) -> torch.Tensor:
        """
        Returns the results of the computations in the batch.

        Returns:
            list[float]: A list of results corresponding to the computations in the batch. We guarantee all results are integers if they are valid.
                         Invalid results are represented as `float('nan')`, which is why we use a float type for the results.
        """
        if self.result_cache is None:
            self.result_cache = torch.tensor([GPTBaseInferenceBatch._result_extractor_fn(computation) for computation in self.computations])
        return self.result_cache

    def __len__(self) -> int:
        return len(self.computations)

    def __getitem__(self, index: int) -> tuple[str, float]:
        """
        Returns the computation and its result at the given index. Normal accessor function with x[index] where x is an instance of this class.

        Args:
            index (int): The index of the sample in the batch to retrieve.

        Returns:
            tuple[str, float]: A tuple containing the computation and its result.
        """
        return self.computations[index], self.results[index]

    @staticmethod
    def _result_extractor_fn(computation: str) -> float:
        result_parts = computation.split("Result:")
        if len(result_parts) == 2:
            ans_str: str = "".join(filter(lambda x: x in ASCII_NUMBERS, result_parts[1].strip()))

            if len(ans_str) > 0:
                return int(ans_str) & ((1<<63) - 1)

        # no result found
        return torch.nan

    def __str__(self):
        return "(" + __class__.__name__ + ")[\n" + "\n".join([f"{computation}: {result}" for computation, result in self]) + "]"

class GPTBaseWrapper:
    def __init__(self, gptbase_model: GPTStyleBaseline | str, device: torch.device, tokenizer: GPTBaseTokenizer, max_inference_length: int | None = None):
        """
        Wrapper class for inference on trained GPTStyleBaseline models

        Args:
            gptbase_model (GPTStyleBaseline | str): The GPTStyleBaseline model or its path.
            device (torch.device): The device to run the model on.
            tokenizer (GPTBaseTokenizer): The tokenizer to use for encoding and decoding.
            max_inference_length (int | None, optional): The maximum inference length. Defaults to None. If this is None, the model will use its currently set maximum inference length.
        """
        if isinstance(gptbase_model, GPTStyleBaseline):
            self.model = gptbase_model
        else:
            self.model = GPTStyleBaseline.load_from_path(gptbase_model)

        self.tokenizer = tokenizer
        self.device = device

        if max_inference_length is not None:
            self.model.set_max_inference_steps(max_inference_length)

        self.model.to(device)
        self.logger = getLogger(__name__)
        self.model.eval()

    def compute_from_databatch(self, x: list[str]) -> GPTBaseInferenceBatch:
        """
        Compute the inference batch for a list of strings as provided by one of the compatible datasets (the ones registered in train_gptbase.py).

        Args:
            x (list[str]): The list of strings to compute.

        Returns:
            GPTBaseInferenceBatch: The inference batch.
        """
        wastrain = self.model.training
        self.model.eval()
        try:
            tokenized_batch: dict[str, torch.Tensor] = self.tokenizer.encode_batch(x, inference_mode=True)
            computations: torch.Tensor = self.model.batch_inference(tokenized_batch["input_ids"], tokenized_batch["attention_mask"])
            return GPTBaseInferenceBatch(self.tokenizer.strip_decode(computations))
        finally:
            if wastrain:
                self.model.train()
