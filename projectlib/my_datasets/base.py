import os
import torch
import numpy as np

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from dataclasses import dataclass


DATASETS_BASE_DIR = "datasets/"

TOKENIZER_MAX_LENGTH = 20
RANDOM_SEED = 0


@dataclass
class GenerationSpec:
    """Class that specifies the parameters for the dataset generation."""

    size: int  # Number of samples to generate
    low: int  # Lower bound for random values
    high: int  # Upper bound for random values


class GeneratedDataset(Dataset, ABC):
    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer = None,
        regenerate: bool = False,
        generation_spec: GenerationSpec = None,
        max_length: int = TOKENIZER_MAX_LENGTH,
    ):
        super().__init__()

        # fix random seed for reproducibility
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

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

    def __getitem__(self, idx: int):
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
