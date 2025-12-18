import os
import torch
import numpy as np
import random
import math
import sys

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

    low: int                # Lower bound for random values
    high: int               # Upper bound for random values
    eval_size: int          # Number of evaluation samples to generate
    test_size: int = 0      # Number of test samples to generate
    train_size: int = 0     # Number of training samples to generate

    @staticmethod
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
        generation_spec: GenerationSpec | None,
        tokenizer: Optional[TokenizerType] = None,
        regenerate: bool = True,
        max_length: int = TOKENIZER_MAX_LENGTH,
        split: Split = Split.EVAL,
        seed: Optional[int] = None,
        disallow_op_permutations: bool = False
    ):
        super().__init__()

        # fix random seed for reproducibility, the train flag makes sure that train and eval sets get different random seeds
        seed = seed if seed else RANDOM_SEED
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.tokenizer = tokenizer
        self.max_length = max_length

        if not os.path.exists(DATASETS_BASE_DIR):
            os.makedirs(DATASETS_BASE_DIR)

        if os.path.exists(path) and not regenerate:
            saved = torch.load(path)
            self.data = saved["data"]
            self.labels = saved["labels"]
        elif generation_spec:
            numbers = GeneratedDataset._sample_numbers(generation_spec, disallow_op_permutations)

            e_s = generation_spec.eval_size
            et_s = e_s + generation_spec.test_size

            self.eval_nums = numbers[: e_s]
            self.test_nums = numbers[e_s : et_s]
            self.train_nums = numbers[et_s :]

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

    @staticmethod
    def _sample_numbers(spec: GenerationSpec, disallow_permutations: bool) -> list[tuple[int, int]]:
        """
        Samples non-repeating tuples of numbers within the given range.
        Args:
            spec (GenerationSpec): The specification for the generation.
            disallow_permutations (bool): Whether to disallow permutations of the operands.

        Warning: Sampling both (x, y) and (y, x) is allowed for x != y unless disallow_permutations is True.
        """
        span = spec.high - spec.low # hi is exclusive

        if disallow_permutations:
            total_samples = span * (span - 1) // 2
        else:
            total_samples = span * span

        # we must ensure that the sample space is supportable
        if(total_samples >= sys.maxsize):
            indices = GeneratedDataset._slow_sample(total_samples, spec.train_size + spec.test_size + spec.eval_size)
        else:
            indices = random.sample(range(total_samples), spec.train_size + spec.test_size + spec.eval_size)
        values: list[tuple[int, int]] = []

        # convert indices to (i, j) pairs
        if disallow_permutations:
            for k in indices:
                row = (math.isqrt(8 * k + 1) - 1) // 2
                col = k - row * (row + 1) // 2
                values.append((spec.low + row, spec.low + col))

            # double check validity of generation:  make sure that overlapping values are symmetric in values and v2
            v2 = [(j, i) for i, j in values]
            inters = set(values).intersection(set(v2))
            for i, j in inters:
                assert i == j, f"Duplicate values found: {(i, j), (j, i)}"
        else:
            values = [(spec.low + i // span, spec.low + i % span) for i in indices]

        assert len(values) == len(set(values)), f"Duplicate values found: {values}"
        return values

    @staticmethod
    def _slow_sample(total_samples: int, sample_size: int) -> list[int]:
        samples: set[int] = set([])

        while len(samples) < sample_size:
            sample = random.randint(0, total_samples - 1)
            if sample not in samples:
                samples.add(sample)

        out: list[int] = list(samples)
        random.shuffle(out)
        return out
