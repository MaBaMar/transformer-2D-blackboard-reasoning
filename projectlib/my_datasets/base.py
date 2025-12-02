import os
import torch
import numpy as np

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from abc import ABC, abstractmethod
from typing import Optional, Union, TypeAlias, Any
from dataclasses import dataclass
from enum import Enum



DATASETS_BASE_DIR = "datasets/"

TOKENIZER_MAX_LENGTH = 20
RANDOM_SEED = 0

TokenizerType: TypeAlias = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]



class Split(Enum):
    EVAL = 1
    TEST = 2
    TRAIN = 3

    def size(self, spec: "GenerationSpec"):
        match self:
            case Split.EVAL:
                return spec.eval_size
            case Split.TEST:
                return spec.test_size
            case Split.TRAIN:
                return spec.train_size



@dataclass
class GenerationSpec:
    """Class that specifies the parameters for the dataset generation."""

    eval_size: int          # Number of evaluation samples to generate
    test_size: int = 0      # Number of test samples to generate
    train_size: int = 0     # Number of training samples to generate
    low: int                # Lower bound for random values
    high: int               # Upper bound for random values

    def digits(eval_size: int, digits: int, test_size: int = 0, train_size: int = 0) -> "GenerationSpec":
        """
        Alternate constructor that sets low/high based on number of digits.
        """
        return GenerationSpec(
            eval_size=eval_size,
            test_size=test_size, 
            train_size=train_size,
            low=1, 
            high=10**digits
        )


class GeneratedDataset(Dataset, ABC):
    def __init__(
        self,
        path: str,
        tokenizer: Optional[TokenizerType] = None,
        regenerate: bool = False,
        generation_spec: Optional[GenerationSpec] = None,
        max_length: int = TOKENIZER_MAX_LENGTH,
        split: Split = Split.EVAL,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # fix random seed for reproducibility, the train flag makes sure that train and eval sets get different random seeds
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
            data, labels = self.__generate__(generation_spec, split)
            torch.save({"data": data, "labels": labels}, path)
            self.data = data
            self.labels = labels
        else:
            raise Exception("Dataset generation failed!")

    @abstractmethod
    def __generate__(self, spec: GenerationSpec, split: Split = Split.EVAL):
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
