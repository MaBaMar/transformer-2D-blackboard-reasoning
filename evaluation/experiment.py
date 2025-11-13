import argparse
import torch
import wandb
import re

import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from projectlib.my_datasets import *



MODEL_PATHS = {
    "Llama-13B": "TheBloke/LLaMA-13b-GGUF",
}



#
#   Helper functions
#


def load_dataset(task: str, size: int, digits: int) -> GeneratedDataset:
    low = 10**(digits - 1)
    high = 10**(digits)
    spec = GenerationSpec(size, low, high)

    if task == "addition":
        return AdditionDataset(
            train=False,
            regenerate=True,
            generation_spec=spec,
        )
    elif task == "scratchpad":
        return ScratchpadDataset(
            train=False,
            regenerate=True,
            generation_spec=spec,
        )
    elif task == "blackboard":
        raise NotImplementedError("Implement blackboard!")
    else:
        raise TypeError("Unsupported task!")
    


def check_prediction(prediction: str, label: str) -> int:
    result_true = extract_label_number(label)
    result_pred = extract_label_number(prediction)

    print(f"result_true: {result_true} and result_pred: {result_pred}")

    if int(result_true) == int(result_pred):
        return 1

    return 0
    


def extract_label_number(label: str) -> int | None:
    """
    Extract the final numeric result from a dataset label string.
    Returns an int or None if no match.
    """
    # Look for the last 'Result:' followed by digits and spaces
    m = re.findall(r"Result:\s*([0-9 ]+)", label)
    raw = m[-1].strip().replace(" ", "")
    label = int(raw) if raw.isdigit() else None
    return label


#
#   Main experiment function
#


def experiment(
        name: str,
        model_name: str,
        task: str,
        size: int,
        digits: int,
        seed: int,
        logging: str = "local",
    ):

    wandb.init(
        entity="blackboard-reasoning",
        project="blackboard-reasoning",
        config={
            "name": name,
            "model": model_name,
            "path_to_model": MODEL_PATHS[model_name],
            "task": task,
            "size": size,
            "digits": digits,
            "seed": seed,
        },
        mode="online" if logging == "wandb" else "offline"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = 0 if torch.cuda.is_available() else -1 # 0 for GPU, -1 for CPU

    #
    #   Load the model and the dataset
    #

    print(f"Evaluating {model_name} on {task} with {digits}-digits\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_name])
    model = AutoModel.from_pretrained(
        MODEL_PATHS[model_name], 
        dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    dataset = load_dataset(task, size, digits)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    #
    #   Evaluate the performance on the dataset
    #

    correct = 0.0

    for element in tqdm(dataloader):
        input_text = element["input"]
        label = element["label"]

        input = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **input,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
            )

        prediction = tokenizer.decode(output, skip_special_tokens=True).strip()

        correct += check_prediction(prediction, label)

    acc = correct / size

    wandb.log({
        "accuracy": acc,
    })

    wandb.finish()


    

def main(args):
    experiment(
        name=args.name,
        model_name=args.model_name,
        task=args.task,
        digits=args.digits,
        seed=args.seed,
        logging=args.logging,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--digits", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    args, _ = parser.parse_known_args()
    main(args)
