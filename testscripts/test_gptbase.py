
import torch
from torch.utils.data import DataLoader

import numpy as np
import os
from logging import getLogger

from projectlib.my_datasets import Split, GenerationSpec
from src.training.train_gptbase import GPTBaseTokenizer, GPTStyleBaseline, _DATA_T_REGISTRY
from src.evaluation.gptbase_wrapper import GPTBaseWrapper

if __name__ == "__main__":

    # SET parameters as used during training
    seed = 0
    digits = 4
    eval_size = 1000
    test_size = 1000
    train_size = 10000
    batch_size = 32

    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_variant = "cot"

    trainlogger = getLogger(__name__ + os.path.basename(__file__))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = GenerationSpec.digits(
        digits=digits,
        eval_size=eval_size,
        test_size=test_size,
        train_size=train_size,
    )

    dataset_test = _DATA_T_REGISTRY[dataset_variant](
        regenerate=True,
        split=Split.EVAL,
        seed=seed,
        generation_spec=spec,
    )

    tokenizer: GPTBaseTokenizer = GPTBaseTokenizer(torch.device(device))

    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model: GPTStyleBaseline = GPTStyleBaseline.load_from_path("./models/cot_test_d4_s0.pt")

    sample_inspect_count = 5

    for batch in test_loader:
        print("----------------[SAMPLE START]----------------")
        print("Original Prompt:", batch["input"][0])
        tokenized_batch = tokenizer.encode_batch(batch["input"], True)
        y_hat = model.batch_inference(tokenized_batch["input_ids"], tokenized_batch["attention_mask"])
        print("[Model Inference Output]:")
        print(">  " + tokenizer.strip_decode(y_hat)[0].replace("\n", "\n>  "))

        print("Expected Answer:", batch["label"][0])
        print("----------------[SAMPLE END]----------------\n\n\n")

        if sample_inspect_count > 1:
            sample_inspect_count -= 1
        else:
            break

    wrapper = GPTBaseWrapper(model, model.device, tokenizer)

    test_batch = next(iter(test_loader))

    xs = wrapper.compute_from_databatch(test_batch["input"])
    print(xs)
    print(xs.results, test_batch["label"])
    print("Mean Accuracy:", (xs.results == test_batch["label"]).float().mean().item())
