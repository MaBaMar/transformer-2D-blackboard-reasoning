"""
Evaluation script for error correction models.

Evaluates models on three tasks:
1. Normal progression - standard state → next state accuracy
2. Model's own errors - collect mistakes the model makes, see if it can correct them
3. Synthetic errors - corrupt samples the same way as training, test correction
"""
import argparse
import torch
import wandb
import random

import numpy as np

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.eogar import EOgar
from projectlib.my_datasets import *
from projectlib.my_datasets.collators import collate_bb_state_state, make_collator_with_args
from projectlib.my_datasets.blackboards import bb_prettyprint






class ErrorDataset(Dataset):
    def __init__(self, error_list: list):
        self.error_list = error_list
    def __len__(self) -> int:
        return len(self.error_list)
    def __getitem__(self, idx: int) -> object:
        return self.error_list[idx]


def construct_synthetic_errors(dataset: Dataset, num_errors: int, bb_width: int, seed: int = 0) -> list:
    random.seed(seed)
    
    DIGIT_OFFSET = 4
    CARRY_ROW = 2
    RESULT_ROW = 4
    
    synthetic_errors = []
    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)
    
    for idx in dataset_indices:
        if len(synthetic_errors) >= num_errors:
            break
            
        item = dataset[idx]
        x_dict, y_dict = item
        
        y_tokens = y_dict["tokens"].clone()
        
        carry_col = None
        for col in range(bb_width):
            token = y_tokens[CARRY_ROW, col].item()
            if DIGIT_OFFSET <= token <= DIGIT_OFFSET + 9:
                carry_col = col
                break
        
        result_col = None
        for col in range(bb_width):
            token = y_tokens[RESULT_ROW, col].item()
            if DIGIT_OFFSET <= token <= DIGIT_OFFSET + 9:
                result_col = col
                break
        
        if carry_col is None or result_col is None:
            continue
        
        corrupted_y_tokens = y_tokens.clone()
        
        new_carry = random.randint(0, 1)
        corrupted_y_tokens[CARRY_ROW, carry_col] = DIGIT_OFFSET + new_carry
        
        correct_digit = y_tokens[RESULT_ROW, result_col].item() - DIGIT_OFFSET
        wrong_digits = [d for d in range(10) if d != correct_digit]
        new_digit = random.choice(wrong_digits)
        corrupted_y_tokens[RESULT_ROW, result_col] = DIGIT_OFFSET + new_digit
        
        corrupted_y_dict = {
            "tokens": corrupted_y_tokens,
            "pos_row": y_dict["pos_row"].clone(),
            "pos_col": y_dict["pos_col"].clone(),
        }
        correct_y_dict = {
            "tokens": y_tokens,
            "pos_row": y_dict["pos_row"].clone(),
            "pos_col": y_dict["pos_col"].clone(),
        }
        
        synthetic_errors.append((corrupted_y_dict, correct_y_dict))
    
    return synthetic_errors


def collect_model_errors(model: EOgar, dataset: Dataset, collate_fn: callable, batch_size: int, bb_height: int, bb_width: int) -> list:
    model.eval()
    collected_errors = []
    
    EOS_TOKEN_ID = 1  
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits, _ = model(x_batch, y_batch)
            preds = logits.argmax(dim=-1)
            
            target_tokens = y_batch[0]
            y_pos_row = y_batch[1]
            y_pos_col = y_batch[2]
            
            if target_tokens.shape != preds.shape:
                continue
            
            is_correct = (preds == target_tokens).all(dim=1)
            target_has_eos = (target_tokens == EOS_TOKEN_ID).any(dim=1)
            
            for i, correct in enumerate(is_correct):
                if correct or target_has_eos[i]:
                    continue
                    
                wrong_pred_tokens = preds[i].cpu().view(bb_height, bb_width).clone()
                correct_target_tokens = target_tokens[i].cpu().view(bb_height, bb_width).clone()
                    
                wrong_pred_dict = {
                    "tokens": wrong_pred_tokens,
                    "pos_row": y_pos_row[i].cpu(),
                    "pos_col": y_pos_col[i].cpu(),
                }
                correct_target_dict = {
                    "tokens": correct_target_tokens,
                    "pos_row": y_pos_row[i].cpu(),
                    "pos_col": y_pos_col[i].cpu(),
                }
                collected_errors.append((wrong_pred_dict, correct_target_dict))
    
    return collected_errors


def evaluate_normal_progression(model: EOgar, dataset: Dataset, collate_fn: callable, batch_size: int) -> tuple:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    total_correct_states = 0
    total_correct_tokens = 0
    total_tokens = 0
    total_samples = 0
    
    EOS_TOKEN_ID = 1 

    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits, _ = model(x_batch, y_batch)
            preds = logits.argmax(dim=-1)
            target_tokens = y_batch[0]
            
            if target_tokens.shape != preds.shape:
                continue
            
            exact_match = (preds == target_tokens).all(dim=1)
            both_have_eos = (preds == EOS_TOKEN_ID).any(dim=1) & (target_tokens == EOS_TOKEN_ID).any(dim=1)
            state_correct = exact_match | both_have_eos
            total_correct_states += state_correct.sum().item()
            
            target_has_eos = (target_tokens == EOS_TOKEN_ID).any(dim=1)
            for i in range(target_tokens.shape[0]):
                if not target_has_eos[i]:
                    token_correct = (preds[i] == target_tokens[i]).sum().item()
                    total_correct_tokens += token_correct
                    total_tokens += target_tokens[i].numel()
            
            total_samples += target_tokens.shape[0]
    
    state_acc = total_correct_states / total_samples if total_samples > 0 else 0.0
    token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return state_acc, token_acc, total_samples


def evaluate_error_correction(model: EOgar, error_samples: list, collate_fn: callable, batch_size: int, task_name: str = "errors") -> tuple:
    if not error_samples:
        print(f"[{task_name}] No error samples to evaluate!")
        return 0.0, 0.0, 0
    
    model.eval()
    error_dataset = ErrorDataset(error_samples)
    loader = DataLoader(error_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    total_correct_states = 0
    total_correct_tokens = 0
    total_tokens = 0
    total_samples = 0
    
    EOS_TOKEN_ID = 1  
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits, _ = model(x_batch, y_batch)
            preds = logits.argmax(dim=-1)
            target_tokens = y_batch[0]
            
            if target_tokens.shape != preds.shape:
                continue
            
            exact_match = (preds == target_tokens).all(dim=1)
            both_have_eos = (preds == EOS_TOKEN_ID).any(dim=1) & (target_tokens == EOS_TOKEN_ID).any(dim=1)
            state_correct = exact_match | both_have_eos
            total_correct_states += state_correct.sum().item()
            
            target_has_eos = (target_tokens == EOS_TOKEN_ID).any(dim=1)
            for i in range(target_tokens.shape[0]):
                if not target_has_eos[i]:
                    token_correct = (preds[i] == target_tokens[i]).sum().item()
                    total_correct_tokens += token_correct
                    total_tokens += target_tokens[i].numel()
            
            total_samples += target_tokens.shape[0]
    
    state_acc = total_correct_states / total_samples if total_samples > 0 else 0.0
    token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return state_acc, token_acc, total_samples


def experiment(
        name: str,
        model_name: str,
        model_path: str,
        size: int,
        batch_size: int,
        digits: int,
        bb_spec: BlackboardSpec,
        seed: int,
        logging: str = "local",
    ):

    wandb.init(
        name=name,
        entity="blackboard-reasoning",
        project="blackboard-reasoning",
        config={
            "model": model_name,
            "size": size,
            "batch_size": batch_size,
            "bb_spec": {
                "height": bb_spec.height,
                "width": bb_spec.width,
                "randomize_position": bb_spec.randomize_position,
                "operation": str(bb_spec.operation),
            },
            "digits": digits,
            "seed": seed,
        },
        mode="online" if logging == "wandb" else "offline"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"="*60)
    print(f"Evaluating {model_name} on {digits}-digit problems")
    print(f"Model path: {model_path}")
    print(f"="*60)

    # Load model
    model = EOgar.load_from_path(model_path).to(device)
    model.eval()

    # Create dataset
    spec = GenerationSpec(
        low=10**(digits - 1),
        high=10**(digits),
        eval_size=size,
        test_size=size,
        train_size=size,
    )

    dataset = TokenizedBlackboardDataset(
        regenerate=True,
        split=Split.TEST,
        seed=seed,
        generation_spec=spec,
        blackboard_spec=bb_spec,
    )

    pad_id = dataset.bb_2D_tokenizer.pad_id
    collate_fn = make_collator_with_args(collate_bb_state_state, pad_token_id=pad_id, device=device)

    print(f"\nDataset size: {len(dataset)} samples")

    print(f"\n{'='*60}")
    print("Task 1: NORMAL PROGRESSION (state → next state)")
    print(f"{'='*60}")
    
    normal_state_acc, normal_token_acc, normal_samples = evaluate_normal_progression(
        model=model,
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size
    )
    
    print(f"  State Accuracy: {normal_state_acc*100:.2f}%")
    print(f"  Token Accuracy: {normal_token_acc*100:.2f}%")
    print(f"  Samples: {normal_samples}")

    print(f"\n{'='*60}")
    print("Task 2: MODEL'S OWN ERRORS (can it correct its mistakes?)")
    print(f"{'='*60}")
    
    model_errors = collect_model_errors(
        model=model,
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        bb_height=bb_spec.height,
        bb_width=bb_spec.width
    )
    
    print(f"  Model made {len(model_errors)} errors out of {len(dataset)} samples ({len(model_errors)/len(dataset)*100:.1f}%)")
    
    if model_errors:
        own_error_state_acc, own_error_token_acc, own_error_samples = evaluate_error_correction(
            model=model,
            error_samples=model_errors,
            collate_fn=collate_fn,
            batch_size=batch_size,
            bb_height=bb_spec.height,
            bb_width=bb_spec.width,
            task_name="own_errors"
        )
        print(f"  Correction State Accuracy: {own_error_state_acc*100:.2f}%")
        print(f"  Correction Token Accuracy: {own_error_token_acc*100:.2f}%")
        
        if len(model_errors) > 0:
            print(f"\n  --- First 3 error samples ---")
            num_examples = min(3, len(model_errors))
            for idx in range(num_examples):
                wrong_dict, correct_dict = model_errors[idx]
                
                with torch.no_grad():
                    x_batch, y_batch = collate_fn([model_errors[idx]])
                    logits, _ = model(x_batch, y_batch)
                    pred = logits.argmax(dim=-1)[0].view(bb_spec.height, bb_spec.width)
                
                print(f"\n  Example {idx+1}:")
                print("  INPUT (model's wrong prediction):")
                bb_prettyprint(wrong_dict["tokens"])
                print("  MODEL OUTPUT (correction attempt):")
                bb_prettyprint(pred)
                print("  TARGET (correct state):")
                bb_prettyprint(correct_dict["tokens"])
    else:
        own_error_state_acc, own_error_token_acc, own_error_samples = 1.0, 1.0, 0
        print("  Model is perfect - no errors to correct!")


    print(f"\n{'='*60}")
    print("Task 3: SYNTHETIC ERRORS (corrupted carry + result digit)")
    print(f"{'='*60}")
    
    synthetic_errors = construct_synthetic_errors(
        dataset=dataset,
        num_errors=len(dataset),
        bb_width=bb_spec.width,
        seed=seed
    )
    
    print(f"  Generated {len(synthetic_errors)} synthetic errors")
    
    synthetic_state_acc, synthetic_token_acc, synthetic_samples = evaluate_error_correction(
        model=model,
        error_samples=synthetic_errors,
        collate_fn=collate_fn,
        batch_size=batch_size,
        task_name="synthetic"
    )
    
    print(f"  Correction State Accuracy: {synthetic_state_acc*100:.2f}%")
    print(f"  Correction Token Accuracy: {synthetic_token_acc*100:.2f}%")
    
    if len(synthetic_errors) > 0:
        print(f"\n  --- First 3 synthetic error samples ---")
        num_examples = min(3, len(synthetic_errors))
        for idx in range(num_examples):
            corrupted_dict, correct_dict = synthetic_errors[idx]
            
            with torch.no_grad():
                x_batch, y_batch = collate_fn([synthetic_errors[idx]])
                logits, _ = model(x_batch, y_batch)
                pred = logits.argmax(dim=-1)[0].view(bb_spec.height, bb_spec.width)
            


    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Normal Progression:    {normal_state_acc*100:.2f}% state, {normal_token_acc*100:.2f}% token")
    print(f"  Own Error Correction:  {own_error_state_acc*100:.2f}% state, {own_error_token_acc*100:.2f}% token ({own_error_samples} errors)")
    print(f"  Synthetic Correction:  {synthetic_state_acc*100:.2f}% state, {synthetic_token_acc*100:.2f}% token ({synthetic_samples} samples)")

    # Log to wandb
    wandb.log({
        "normal_state_acc": normal_state_acc,
        "normal_token_acc": normal_token_acc,
        "own_error_state_acc": own_error_state_acc,
        "own_error_token_acc": own_error_token_acc,
        "own_error_count": len(model_errors),
        "synthetic_state_acc": synthetic_state_acc,
        "synthetic_token_acc": synthetic_token_acc,
    })

    wandb.finish()


def main(args):
    bb_op = Addition() if args.operation == "addition" else None

    experiment(
        name=args.name,
        model_name=args.model_name,
        model_path=args.model_path,
        digits=args.digits,
        size=args.size,
        batch_size=args.batch_size,
        bb_spec=BlackboardSpec(args.height, args.width, args.randomize_position, bb_op),
        seed=args.seed,
        logging=args.logging,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--digits", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--randomize_position", action="store_true")
    parser.add_argument("--operation", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    args, _ = parser.parse_known_args()
    main(args)
