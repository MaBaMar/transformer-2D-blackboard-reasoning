import os
import torch
import numpy as np

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from abc import ABC, abstractmethod
from typing import Optional, Union, TypeAlias, Any
from dataclasses import dataclass

DATASETS_BASE_DIR = "datasets/"

TOKENIZER_MAX_LENGTH = 20
RANDOM_SEED = 0

TokenizerType: TypeAlias = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

@dataclass
class GenerationSpec:
    """Class that specifies the parameters for the dataset generation."""

    size: int  # Number of samples to generate
    low: int  # Lower bound for random values
    high: int  # Upper bound for random values

    def digits(size: int, digits: int) -> "GenerationSpec":
        """
        Alternate constructor that sets low/high based on number of digits.
        """
        return GenerationSpec(size, low=1, high=10**digits)


class GeneratedDataset(Dataset, ABC):
    def __init__(
        self,
        path: str,
        tokenizer: Optional[TokenizerType] = None,
        regenerate: bool = False,
        generation_spec: Optional[GenerationSpec] = None,
        max_length: int = TOKENIZER_MAX_LENGTH,
        train: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # fix random seed for reproducibility, the train flag makes sure that train and eval sets get different random seeds
        seed = (seed or RANDOM_SEED) + (not train)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.tokenizer = tokenizer
        self.max_length = max_length

        if not os.path.exists(DATASETS_BASE_DIR):
            os.makedirs(DATASETS_BASE_DIR)

        if os.path.exists(path) and not regenerate:
            saved = torch.load(path)
            self.data = saved["data"]
            self.labels = saved["labels"]
        elif generation_spec:
            data, labels = self.__generate__(generation_spec)
            torch.save({"data": data, "labels": labels}, path)
            self.data = data
            self.labels = labels
        else:
            raise Exception("Dataset generation failed!")

    @abstractmethod
    def __generate__(self, spec: GenerationSpec):
        raise NotImplementedError("Subclasses must implement __generate__ method")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Any:
        input = self.data[idx]
        label = self.labels[idx]

        if self.tokenizer:
            input_encoded = self.tokenizer(
                input,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            label_encoded = self.tokenizer(
                label,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids": input_encoded["input_ids"].squeeze(0),
                "attention_mask": input_encoded["attention_mask"].squeeze(0),
                "labels": label_encoded["input_ids"].squeeze(0),
            }

        return {"input": input, "label": label}
