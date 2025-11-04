import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from my_datasets.base import GeneratedDataset, GenerationSpec
        


EVAL_PATH = "datasets/additions_eval.pt"
TRAIN_PATH = "datasets/additions_train.pt"



BASE_SPEC = GenerationSpec(10, 10, 20)



class AdditionDataset(GeneratedDataset):
    def __init__(self, path: str=None, tokenizer: AutoTokenizer=None, train: bool=True, regenerate: bool=False, generation_spec: GenerationSpec=BASE_SPEC):
        path = path if path else (TRAIN_PATH if train else EVAL_PATH)
        super().__init__(
            path=path, 
            tokenizer=tokenizer,
            regenerate=regenerate,
            generation_spec=generation_spec,
        )

    
    def __generate__(self, spec: GenerationSpec):
        inputs = []
        labels = []

        for _ in range(spec.size):
            a = torch.randint(spec.low, spec.high, (1,)).item()
            b = torch.randint(spec.low, spec.high, (1,)).item()
            
            inputs.append(f"{a} + {b}")
            labels.append(f"{a + b}")

        return inputs, labels
