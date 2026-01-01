
import torch
from torch.utils.data import DataLoader

import numpy as np
import os
from logging import getLogger
from tqdm import tqdm

from projectlib.my_datasets import Split, GenerationSpec
from src.training.train_gptbase import GPTBaseTokenizer, GPTStyleBaseline, _DATA_T_REGISTRY
from src.evaluation.gptbase_wrapper import GPTBaseInferenceBatch, GPTBaseWrapper

if __name__ == "__main__":

    # SET parameters as used during training
    seed = 0
    digits = 12
    eval_size = 1000
    test_size = 1000
    train_size = 8000
    batch_size = 12

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

    model: GPTStyleBaseline = GPTStyleBaseline.load_from_path("./models/cot_test-bound_weights_d12_s0.pt")
    model.eval()

    sample_inspect_count = 2

    for batch in test_loader:
        print("----------------[SAMPLE START]----------------")
        print("Original Prompt:", batch["input"][0])
        tokenized_batch = tokenizer.encode_batch(batch["input"], True)
        y_hat = model.batch_inference(tokenized_batch["input_ids"], tokenized_batch["attention_mask"])
        print("[Model Inference Output]:")
        print(">  " + (v:=tokenizer.strip_decode(y_hat))[0].replace("\n", "\n>  "))

        print("Expected Answer:", batch["label"][0])
        print("Extracted Result:", (b:=GPTBaseInferenceBatch(v)).results[0])
        print("Batch accuracy:", (b.results == batch["label"]).float().mean().item())
        print("----------------[SAMPLE END]----------------\n\n\n")

        if sample_inspect_count > 1:
            sample_inspect_count -= 1
        else:
            break

    wrapper = GPTBaseWrapper(model, model.device, tokenizer)


    test_batch = next(iter(test_loader))

    # ------------------------------------------------
    # testing left-padding invariance during inference.
    # uncomment this and set batch size to 1 to test this
    # ------------------------------------------------
    #
    # inpt = test_batch["input"]
    # for i in range(10):
    #     print(f"adding {i} padding")
    #     ipt2 = [(i)*"<pad>" + xel for xel in inpt]
    #     print(wrapper.compute_from_databatch(ipt2))
    acc = 0
    print("Computing accuracy")
    progress_bar = tqdm(test_loader)
    for test_batch in progress_bar:
        xs = wrapper.compute_from_databatch(test_batch["input"])
        print("wrong sample:", xs.results[xs.results != test_batch["label"]], test_batch["label"][xs.results != test_batch["label"]])
        batch_acc = (xs.results == test_batch["label"]).float().mean().item()
        acc += batch_acc
        progress_bar.set_postfix_str(f"batch_accuracy: {batch_acc}")

    print(f"Mean accuracy: {acc / len(test_loader)}")
