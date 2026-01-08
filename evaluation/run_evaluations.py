import argparse
import torch
import wandb
import re

import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.evaluation.gptbase_wrapper import GPTBaseWrapper
from src.models.eogar import EOgar
from projectlib.my_datasets import *
from projectlib.my_datasets.collators import collate_bb_state_int, make_collator_with_args
from projectlib.my_datasets.blackboards import CarryOperation
from src.evaluation.bb_chain_wrapper import BBChainReasoner, chainlist_to_results
from src.models.gptbase import GPTBaseTokenizer, GPTStyleBaseline, _DATA_T_REGISTRY
from src.training.train_gptbase import _SP_OP_REGISTRY


BB_OPERATION: dict[str, CarryOperation] = {
    "add": Addition(),
    "sub": Subtraction(),
}



#
#   Helper functions
#


def setup_model(model_path, digits: int, bb_spec: BlackboardSpec, task: str, device=-1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if task == "basic":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
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

    elif task == "scratchpad" or task == "cot":
        tok = GPTBaseTokenizer(torch.device(device))
        model = GPTStyleBaseline.load_from_path(model_path)
        reasoner = GPTBaseWrapper(model, torch.device(device), tok)

        return reasoner, tok

    elif task == "blackboard-1d":
        tok = BBVocabTokenizer()
        model = EOgar.load_from_path(model_path)

        reasoner = BBChainReasoner(model, torch.device(device), bb_spec, timeout_iters=2 * digits + 2, tokenizer=tok)

        return reasoner, tok

    elif task == "blackboard-2d":
        tok = BBVocabTokenizer()
        model = EOgar.load_from_path(model_path)

        reasoner = BBChainReasoner(model, torch.device(device), bb_spec, timeout_iters=2 * digits + 2, tokenizer=tok)

        return reasoner, tok

    else:
        raise TypeError("Unsupported task!")



def load_dataset(task: str,
                 size: int,
                 digits: int,
                 bb_height: int,
                 bb_width: int,
                 bb_rand_pos: bool,
                 operation: str,
                 seed: int,
                 batch_size: int = 1) -> DataLoader:
    spec = GenerationSpec(
        low=10**(digits - 1),
        high=10**(digits),
        eval_size=size,
    )

    if task == "basic":
        raise NotImplementedError("Implement basic!")

    elif task == "scratchpad" or task == "cot":
        if operation == "mixed":
            ds = []

            for op in _SP_OP_REGISTRY.values():
                ds.append(
                    _DATA_T_REGISTRY[task](
                        split=Split.EVAL,
                        seed=seed,
                        generation_spec=spec,
                        operand=_SP_OP_REGISTRY[operation],
                    )
                )

            dataset = ConcatDataset(ds)

        else:
            dataset = _DATA_T_REGISTRY[task](
                split=Split.EVAL,
                seed=seed,
                generation_spec=spec,
                operand=_SP_OP_REGISTRY[operation],
            )

        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    elif task == "blackboard-1d" or task == "blackboard-2d":
        if operation == "mixed":
            ds = []

            for op in BB_OPERATION.values():
                bb_spec = BlackboardSpec(
                    height=bb_height,
                    width=bb_width,
                    randomize_position=bb_rand_pos,
                    operation=op,
                )

                ds.append(
                    TokenizedBlackboardDataset(
                        split=Split.EVAL,
                        seed=seed,
                        generation_spec=spec,
                        blackboard_spec=bb_spec,
                    )
                )

            dataset = ConcatDataset(ds)

            pad_id = ds[0].bb_2D_tokenizer.pad_id

        else:
            bb_spec = BlackboardSpec(
                height=bb_height,
                width=bb_width,
                randomize_position=bb_rand_pos,
                operation=BB_OPERATION[operation],
            )

            dataset = TokenizedBlackboardDataset(
                split=Split.EVAL,
                seed=seed,
                generation_spec=spec,
                blackboard_spec=bb_spec,
            )

            pad_id = dataset.bb_2D_tokenizer.pad_id

        device = "cuda" if torch.cuda.is_available() else "cpu"

        collate_fn = make_collator_with_args(collate_bb_state_int, pad_token_id=pad_id, device=device)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    else:
        raise TypeError("Unsupported task!")



def ask(input, task, pipe, tok):
    if task == "basic":
        raise NotImplementedError("Implement basic!")

    elif task == "scratchpad" or task == "cot":
        pred = pipe.compute_from_databatch(input).results

    elif task == "blackboard-2d" or task == "blackboard-1d":
        out = pipe.compute_from_databatch(input)
        pred = chainlist_to_results(out)

    else:
        raise TypeError("Unsupported task!")

    return pred



def check_prediction(prediction, label, task) -> int:
    if task == "basic":
        raise NotImplementedError("Implement basic!")

    elif task in ["scratchpad", "cot", "blackboard-2d", "blackboard-1d"]:
        # we need to do sum, as the last batch might not be full (we have drop_last = False in the dataloader)
        return (prediction == label).float().sum()

    else:
        raise TypeError("Unsupported task!")



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
        task: str,
        size: int,
        batch_size: int,
        digits: int,
        bb_height: int,
        bb_width: int,
        bb_rand_pos: bool,
        operation: str,
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
            "batch_size": batch_size,
            "bb_spec": {
                "height": bb_height,
                "width": bb_width,
                "randomize_position": bb_rand_pos,
                "operation": operation,
            },
            "operation": operation,
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

    print(f"Evaluating {model_name} on {task} with {digits}-digits and randomize_position: {bb_rand_pos}\n")

    bb_spec = BlackboardSpec(bb_height, bb_width, bb_rand_pos, Addition() if operation == "add" else Subtraction())
    pipe, tok = setup_model(
        model_path=model_path,
        digits=digits,
        bb_spec=bb_spec,
        task=task,
        device=device
    )

    dataloader = load_dataset(
        task=task,
        size=size,
        digits=digits,
        bb_height=bb_height,
        bb_width=bb_width,
        bb_rand_pos=bb_rand_pos,
        operation=operation,
        seed=seed,
        batch_size=batch_size
    )

    #
    #   Evaluate the performance on the dataset
    #

    correct = 0.0
    elem_count = 0

    for element in tqdm(dataloader):
        if task in ["blackboard-1d", "blackboard-2d"]:
            input_text = element[0]
            label = element[-1]
        else:
            input_text = element["input"]
            label = element["label"]

        prediction = ask(input_text, task, pipe, tok).to(device)
        label = label.to(device)

        correct += check_prediction(prediction, label, task)
        elem_count += len(label)

    acc = correct / elem_count

    wandb.log({
        "accuracy": acc,
    })

    wandb.finish()



def main(args):
    experiment(
        name=args.name,
        model_name=args.model_name,
        model_path=args.model_path,
        task=args.task,
        digits=args.digits,
        size=args.size,
        batch_size=args.batch_size,
        bb_height=args.height,
        bb_width=args.width,
        bb_rand_pos=args.randomize_position,
        operation=args.operation,
        seed=args.seed,
        logging=args.logging,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--digits", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--randomize_position", action="store_true", default=False)
    parser.add_argument("--operation", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    args, _ = parser.parse_known_args()
    main(args)
