import argparse
import torch
import wandb
import re

import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import EOgar
from projectlib.my_datasets import *



MODEL_PATHS = {
    "Llama-13B": "TheBloke/LLaMA-13b-GGUF",
    "Llama-8B": "meta-llama/Meta-Llama-3-8B",
    "Llama-1B": "meta-llama/Llama-Guard-3-1B",
}



#
#   Helper functions
#


def setup_model(model, digits: int, task: str, device=-1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if task == "basic":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
        )

        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
        )

        return pipe, tok
        
    elif task == "scratchpad":
        raise NotImplementedError("Implement scratchpads!")

    elif task == "blackboard-1d":
        raise NotImplementedError("Implement blackboard-1d!")

    elif task == "blackboard-2d":
        tok = BBVocabTokenizer()
        model = EOgar.load_from_path(model_path)

        reasoner = BBChainReasoner(model, torch.device(device), bb_spec, timeout_iters=digits+2)

        return reasoner, tok

    else:
        raise TypeError("Unsupported task!")



def load_dataset(task: str, size: int, digits: int) -> GeneratedDataset:
    spec = GenerationSpec(
        low=10**(digits - 1),
        high=10**(digits),
        eval_size=size,
    )

    if task == "basic":
        return AdditionDataset(
            split=Split.EVAL,
            seed=seed,
            generation_spec=spec,
        )
    elif task == "scratchpad":
        return ScratchpadDataset(
            split=Split.EVAL,
            seed=seed,
            generation_spec=spec,
        )
    elif task == "blackboard-1d":
        TokenizedBlackboardDataset(
            split=Split.EVAL,
            seed=seed,
            generation_spec=spec,
        )
    elif task == "blackboard-2d":
        TokenizedBlackboardDataset(
            split=Split.EVAL,
            seed=seed,
            generation_spec=spec,
        )
    else:
        raise TypeError("Unsupported task!")
    


def ask(input, task, pipe, tok):
    if task == "basic":
        prompt = (f"Compute the sum. Answer with just the integer. \n Q: {input} = ? A:"    )
        out = pipe(prompt, max_new_tokens=30 ,do_sample=False,
                    truncation=True, pad_token_id=tok.pad_token_id,)[0]["generated_text"]
        added = out[len(prompt):].strip()
        m = re.search(r"-?\d+", added)
        if m:
            pred = int(m.group(0))
        else:
            return None
        
    elif task == "scratch_pad":
        prompt = (f"You are a calculator. Show your work between <scratch> and </scratch>. Then output a single line: \"Result: <number>\" and stop. Your Task including an Example: {input}")
        out = pipe(prompt, max_new_tokens=200 ,do_sample=False,
                    truncation=True, pad_token_id=tok.pad_token_id,)[0]["generated_text"]
        added = out[len(prompt):].strip()
        m = re.findall(r"Result:\s*([0-9 ]+)", added)
        if m:
            raw = m[0].strip()
        else:
            return None
        pred = int(raw.replace(" ", ""))

    elif task == "blackboard-1d":
        raise NotImplementedError("Implement blackboard-1d!")

    elif task == "blackboard-2d":
        out = pipe.compute_from_databatch(input)
        pred = chainlist_to_results(out)

    else:
        raise TypeError("Unsupported task!")

    return pred
    


def check_prediction(prediction: str, label: str, task) -> int:
    if task == "basic":
        result_true = int(label[0])

    elif task == "scratch_pad":
        result_true = extract_label_number(label[0])

    elif task == "blackboard-1d":
        raise NotImplementedError("Implement blackboard-1d!")

    elif task == "blackboard-2d":
        print(prediction)

        print(label)

    else:
        raise TypeError("Unsupported task!")

    print(f"result_true: {result_true} and result_pred: {prediction}")

    if result_true == prediction:
        return 1

    return 0
    


def extract_label_number(label: str) -> int | None:
    """
    Extract the final numeric result from a dataset label string.
    Returns an int or None if no match.
    """
    # Look for the last 'Result:' followed by digits and spaces
    m = re.findall(r"Result:\s*([0-9 ]+)", label)
    if len(m) > 0:
        raw = m[-1].strip().replace(" ", "")
        label = int(raw) if raw.isdigit() else None
    else:
        label = None
    return label


#
#   Main experiment function
#


def experiment(
        name: str,
        model_name: str,
        model_path: str,
        model_location: str,
        task: str,
        size: int,
        digits: int,
        seed: int,
        logging: str = "local",
    ):

    wandb.init(
        name=name,
        entity="blackboard-reasoning",
        project="blackboard-reasoning",
        config={
            "model": model_name,
            "task": task,
            "size": size,
            "digits": digits,
            "seed": seed,
        },
        mode="online" if logging == "wandb" else "offline"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    #   Load the model and the dataset
    #

    print(f"Evaluating {model_name} on {task} with {digits}-digits\n")

    pipe, tok = setup_model(model_path, digits, task, device)

    dataset = load_dataset(task, size, digits)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    #
    #   Evaluate the performance on the dataset
    #

    correct = 0.0

    for element in tqdm(dataloader):
        input_text = element["input"]
        label = element["label"]

        prediction = ask(input_text, task, pipe, tok)

        correct += check_prediction(prediction, label, task)

    acc = correct / size

    wandb.log({
        "accuracy": acc,
    })

    wandb.finish()

    

def main(args):
    experiment(
        name=args.name,
        model_name=args.model_name,
        model_path=args.model_path,
        model_location=args.model_location,
        task=args.task,
        digits=args.digits,
        size=args.size,
        seed=args.seed,
        logging=args.logging,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_location", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--digits", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    args, _ = parser.parse_known_args()
    main(args)
