# minimal training script for EOgar model
import argparse
import os
import random
import re
import torch
import logging

import wandb
import numpy as np

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from tqdm import tqdm

from src.models.eogar import EOgar
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec, Split, BlackboardSpec, Addition, bb_prettyprint
from projectlib.my_datasets.collators import collate_bb_state_state, make_collator_with_args



MODELS_PATH = "./models/"



def compute_accuracy(logits, labels):
    labels = labels[0]
    preds = logits.argmax(dim=-1)

    if labels.shape != preds.shape:
        return 0.0
    
    EOS_TOKEN_ID = 1  # <EOS> token
    
    # State-wise accuracy: exact match OR both contain EOS token
    exact_match = (preds == labels).all(dim=1)
    both_have_eos = (preds == EOS_TOKEN_ID).any(dim=1) & (labels == EOS_TOKEN_ID).any(dim=1)
    state_correct = exact_match | both_have_eos
    output_wise = state_correct.float().mean().item()

    return output_wise


def compute_accuracy_pt(logits, labels):
    labels = labels[0]
    preds = logits.argmax(dim=-1)

    if labels.shape != preds.shape:
        return 0.0
    
    EOS_TOKEN_ID = 1  # <EOS> token
    
    target_has_eos = (labels == EOS_TOKEN_ID).any(dim=1)
    total_correct = 0
    total_tokens = 0
    
    for i in range(labels.shape[0]):
        if not target_has_eos[i]:
            total_correct += (preds[i] == labels[i]).sum().item()
            total_tokens += labels[i].numel()
    
    token_wise = total_correct / total_tokens if total_tokens > 0 else 0.0

    return token_wise


def compute_error_correction_accuracy(model: EOgar, dataset: Dataset, collate_fn: callable, bb_height: int, bb_width: int, seed: int = 0) -> float:
    model.eval()
    
    synthetic_errors = construct_synthetic_errors(
        dataset=dataset,
        num_errors=len(dataset),  
        bb_height=bb_height,
        bb_width=bb_width,
        seed=seed
    )
    
    if not synthetic_errors:
        print("No synthetic errors generated")
        return 0.0

    x_batch, y_batch = collate_fn(synthetic_errors)
    
    with torch.no_grad():
        logits, _ = model(x_batch, y_batch)
        preds = logits.argmax(dim=-1)
        
        target_tokens = y_batch[0]
        input_tokens = x_batch[0]
        
        if target_tokens.shape != preds.shape:
            print(f"Shape mismatch: preds={preds.shape}, target={target_tokens.shape}")
            return 0.0
        
        correct = (preds == target_tokens).all(dim=1).float().mean().item()
    
    return correct


class ErrorDataset(Dataset):
    def __init__(self, error_list: list):
        self.error_list = error_list
    def __len__(self)-> int:
        return len(self.error_list)
    def __getitem__(self, idx: int)-> object:
        return self.error_list[idx]


def construct_train_loader(gold_set: Dataset, collected_errors: list, batch_size: int, collate_fn: callable) -> DataLoader:

    if not collected_errors:
        return DataLoader(gold_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    error_set = ErrorDataset(collected_errors)

    combined_dataset = ConcatDataset([gold_set, error_set])

    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


def construct_synthetic_errors(dataset: Dataset, num_errors: int, bb_height: int, bb_width: int, seed: int = 0) -> list:
    """
    Construct synthetic error samples by corrupting correct targets.
    
    For each sample (x, y) where y is the correct next state:
    - Corrupt y by changing the last carry bit (random 0 or 1)
    - Corrupt y by changing the last result digit (random wrong digit)
    - Return (corrupted_y, y) as error correction training pair
    
    This teaches: corrupted_state → correct_state
    """
    import random
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
        
        
        
        # Find last carry position (row 0, looking for digits 0-1 which are tokens 4-5)
        # Find leftmost (newest) carry position - computation goes right-to-left
        carry_col = None
        for col in range(bb_width):  # Left to right to find newest
            token = y_tokens[CARRY_ROW, col].item()
            if DIGIT_OFFSET <= token <= DIGIT_OFFSET + 9:  # It's a digit
                carry_col = col
                break
        
        # Find leftmost (newest) result digit position
        result_col = None
        for col in range(bb_width):  # Left to right to find newest
            token = y_tokens[RESULT_ROW, col].item()
            if DIGIT_OFFSET <= token <= DIGIT_OFFSET + 9:  # It's a digit
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




def train(
        name: str,
        model_name: str,
        train_size: int,
        test_size: int,
        eval_size: int,
        error_correction: bool,
        error_fraction: float,
        digits: int,
        batch_size: int,
        bb_spec: BlackboardSpec,
        model_dimension: int,
        num_heads_encoder: int,
        n_encoder_blocks: int,
        rope_mode: str,
        learning_rate: float,
        epochs: int,
        seed: int,
        logging: str = "local",
    ):

    wandb.init(
        name=name,
        entity="blackboard-reasoning",
        project="blackboard-reasoning",
        config={
            "model": model_name,
            "train_size": train_size,
            "test_size": test_size,
            "eval_size": eval_size,
            "error_correction": error_correction,
            "error_fraction": error_fraction,
            "digits": digits,
            "batch_size": batch_size,
            "bb_spec": { 
                "height": bb_spec.height, 
                "width": bb_spec.width, 
                "randomize_position": bb_spec.randomize_position, 
                "operation": bb_spec.operation, 
            },
            "model_dimension": model_dimension,
            "num_heads_encoder": num_heads_encoder,
            "n_encoder_blocks": n_encoder_blocks,
            "rope_mode": rope_mode,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "seed": seed,
        },
        mode="online" if logging == "wandb" else "offline"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)


    spec = GenerationSpec.digits(
        digits=digits,
        eval_size=eval_size,
        test_size=test_size,
        train_size=train_size,
    )

    bb_full_dataset_train = TokenizedBlackboardDataset(
        regenerate=True,
        split=Split.TRAIN,
        seed=seed,
        generation_spec=spec,
        blackboard_spec=bb_spec,
    )

    bb_dataset_test = TokenizedBlackboardDataset(
        regenerate=True,
        split=Split.TEST,
        seed=seed,
        generation_spec=spec,
        blackboard_spec=bb_spec,
    )

    pad_id = bb_full_dataset_train.bb_2D_tokenizer.pad_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = bb_full_dataset_train.bb_2D_tokenizer.vocab_size

    collate_fn = make_collator_with_args(collate_bb_state_state, pad_token_id=pad_id, device=device)

    if error_correction:
        # Use full training set 
        bb_dataset_train = bb_full_dataset_train
        print(f"Training dataset size: {len(bb_dataset_train)}")

    else:
        # Initial normal training procedure
        bb_dataset_train = bb_full_dataset_train
        print(f"Training dataset size: {len(bb_dataset_train)}")
        train_loader = DataLoader(
            bb_dataset_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
    
    test_loader = DataLoader(
        bb_dataset_test,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = EOgar(
        vocab_size=vocab_size,
        d_model=model_dimension,
        num_heads_encoder=num_heads_encoder,
        n_encoder_blocks=n_encoder_blocks,
        pad_id=pad_id,
        rope_mode=rope_mode,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params} parameters.")

    for epoch in tqdm(range(epochs)):

        if error_correction:
            # error_fraction determines ratio
            num_errors = int(len(bb_dataset_train) * error_fraction / (1 - error_fraction))
            synthetic_errors = construct_synthetic_errors(
                dataset=bb_dataset_train,
                num_errors=num_errors,
                bb_height=bb_spec.height,
                bb_width=bb_spec.width,
                seed=seed + epoch  
            )
            train_loader = construct_train_loader(gold_set=bb_dataset_train, collected_errors=synthetic_errors, batch_size=batch_size, collate_fn=collate_fn)
            error_pct = len(synthetic_errors) / (len(bb_dataset_train) + len(synthetic_errors)) * 100
            print(f"Epoch {epoch}: {len(bb_dataset_train)} normal + {len(synthetic_errors)} error samples ({error_pct:.1f}% errors).")

        model.train()

        train_acc = 0.0
        train_acc_pt = 0.0
        train_loss = 0.0
        num_batches = 0

        for step, (x_batch, y_batch) in enumerate(train_loader):
            num_batches += 1
            optimizer.zero_grad()

            logits, loss = model(x_batch, y_batch)
    
            loss.backward()
            optimizer.step()

            wandb.log({
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
            })

            batch_acc = compute_accuracy(logits, y_batch)
            train_acc += batch_acc
            train_acc_pt += compute_accuracy_pt(logits, y_batch)
            train_loss += loss.item()

        train_acc /= len(train_loader)
        train_acc_pt /= len(train_loader)
        train_loss /= len(train_loader)
        print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Train Acc PT: {train_acc_pt:.4f}, Train Loss: {train_loss:.4f}")

        model.eval()

        test_acc = 0.0
        test_acc_pt = 0.0
        test_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                logits, loss = model(x_batch, y_batch)

                test_acc += compute_accuracy(logits, y_batch)
                test_acc_pt += compute_accuracy_pt(logits, y_batch)
                test_loss += loss.item()

        test_acc /= len(test_loader)
        test_acc_pt /= len(test_loader)
        test_loss /= len(test_loader)

        error_correction_acc = compute_error_correction_accuracy(
            model=model, 
            dataset=bb_dataset_test, 
            collate_fn=collate_fn, 
            bb_height=bb_spec.height,
            bb_width=bb_spec.width,
            seed=seed + 1000 + epoch,
        ) if error_correction else 0.0

        print(f"Epoch {epoch}: Error Correction Acc: {error_correction_acc:.4f}")
        print(f" normal acc: {test_acc:.4f}, token acc: {test_acc_pt:.4f}")
        wandb.log({
            "epoch": epoch,
            "train_acc": train_acc,
            "train_acc_pt": train_acc_pt,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "test_acc_pt": test_acc_pt,
            "test_loss": test_loss,
            "error_correction_acc": error_correction_acc,
        })

    wandb.finish()

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    save_path = os.path.join(MODELS_PATH, f"{model_name}_d{digits}_s{seed}.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "d_model": model_dimension,
            "num_heads_encoder": num_heads_encoder,
            "n_encoder_blocks": n_encoder_blocks,
            "pad_id": pad_id,
            "rope_mode": rope_mode,
            "epochs": epochs,
            "train_size": train_size,
            "error_correction": error_correction,
            "error_fraction": error_fraction,
            "digits": digits,
            "seed": seed,
        }
    }, save_path)

    print(f"Model saved to {save_path}.")



def main(args):
    bb_spec = BlackboardSpec(
        height=args.bb_height,
        width=args.bb_width,
        randomize_position=args.bb_randomize_position,
        operation=Addition(),
    )

    train(
        name=args.name,
        model_name=args.model_name,
        digits=args.digits,
        train_size=args.train_size,
        test_size=args.test_size,
        eval_size=args.eval_size,
        error_correction=args.error_correction,
        error_fraction=args.error_fraction,
        batch_size=args.batch_size,
        bb_spec=bb_spec,
        model_dimension=args.model_dimension,
        num_heads_encoder=args.num_heads_encoder,
        n_encoder_blocks=args.n_encoder_blocks,
        rope_mode=args.rope_mode,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        logging=args.logging,
    )



if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--digits", type=int, required=True)
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--test_size", type=int, required=True)
    parser.add_argument("--eval_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--bb_height", type=int, required=True)
    parser.add_argument("--bb_width", type=int, required=True)
    parser.add_argument("--bb_randomize_position", action="store_true")
    parser.add_argument("--model_dimension", type=int, required=True)
    parser.add_argument("--num_heads_encoder", type=int, required=True)
    parser.add_argument("--n_encoder_blocks", type=int, required=True)
    parser.add_argument("--rope_mode", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--error_fraction", type=float, required=True)
    parser.add_argument("--error_correction", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    args, _ = parser.parse_known_args()
    main(args)
