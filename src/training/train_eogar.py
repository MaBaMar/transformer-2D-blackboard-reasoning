# minimal training script for EOgar model
import argparse
import os
import torch
import logging

import wandb
import numpy as np

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from tqdm import tqdm

from src.models.eogar import EOgar
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec, Split, BlackboardSpec, Addition
from projectlib.my_datasets.collators import collate_bb_state_state, make_collator_with_args



MODELS_PATH = "./models/"



def compute_accuracy(logits, labels):
    labels = labels[0]
    preds = logits.argmax(dim=-1)

    if labels.shape != preds.shape:
        return 0, 0
    
    output_wise = (preds == labels).all(dim=1).float().mean().item()

    return output_wise


def compute_accuracy_pt(logits, labels):
    labels = labels[0]
    preds = logits.argmax(dim=-1)

    if labels.shape != preds.shape:
        return 0, 0
    
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)

    token_wise = (preds == labels).float().mean().item()

    return token_wise


class ErrorDataset(torch.utils.data.Dataset):
    def __init__(self, error_list):
        self.error_list = error_list
    def __len__(self):
        return len(self.error_list)
    def __getitem__(self, idx):
        return self.error_list[idx]


def construct_train_loader(gold_set, collected_errors, batch_size, collate_fn):

    if not collected_errors:
        return DataLoader(gold_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    error_set = ErrorDataset(collected_errors)

    combined_dataset = torch.utils.data.ConcatDataset([gold_set, error_set])

    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


def collect_error_samples(model, error_pool_set, collate_fn, num_errors, batch_size):
    model.eval()
    collected_errors = []

    loader = DataLoader(error_pool_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    with torch.no_grad():
        for batch_items in loader:
            
            x_collated, y_collated = collate_fn(batch_items)
            
            logits, _ = model(x_collated, y_collated)
            preds = logits.argmax(dim=-1)
            
            target_tokens = y_collated[0]

            if target_tokens.shape != preds.shape:
                continue


            is_correct = (preds == target_tokens).all(dim=1) 

            for i, correct in enumerate(is_correct):
                if not correct:
                    collected_errors.append(batch_items[i])
                    if len(collected_errors) >= num_errors:
                        return collected_errors

    return collected_errors




def train(
        name: str,
        model_name: str,
        train_size: int,
        test_size: int,
        eval_size: int,
        error_correction: bool,
        gold_size: int,
        error_pool_size: int,
        errors_per_epoch: int,
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
            "gold_size": gold_size,
            "error_pool_size": error_pool_size,
            "errors_per_epoch": errors_per_epoch,
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

    if error_correction:
        total_train_size = gold_size + error_pool_size
    else:
        total_train_size = train_size

    spec = GenerationSpec.digits(
        digits=digits,
        eval_size=eval_size,
        test_size=test_size,
        train_size=total_train_size,
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
        train_nums = bb_full_dataset_train.train_nums
        gold_problems = train_nums[:gold_size]
        error_pool_problems = train_nums[gold_size:gold_size + error_pool_size]

        gold_sample_count = 0
        for a, b in gold_problems:
            len_a = int(np.floor(np.log10(a))) + 1 if a > 0 else 1
            len_b = int(np.floor(np.log10(b))) + 1 if b > 0 else 1
            gold_sample_count += max(len_a, len_b) + 2
            
        error_pool_sample_count = 0
        for a, b in error_pool_problems:
            len_a = int(np.floor(np.log10(a))) + 1 if a > 0 else 1
            len_b = int(np.floor(np.log10(b))) + 1 if b > 0 else 1
            error_pool_sample_count += max(len_a, len_b) + 2

        bb_dataset_gold = Subset(bb_full_dataset_train, range(gold_sample_count))
        bb_dataset_error_pool = Subset(bb_full_dataset_train, range(gold_sample_count, gold_sample_count + error_pool_sample_count))


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
            heuristic_scaled_errors_per_epoch = errors_per_epoch * (digits + 2)  # heuristic scaling
            collected_errors = collect_error_samples(model=model, error_pool_set=bb_dataset_error_pool, num_errors=heuristic_scaled_errors_per_epoch, collate_fn=collate_fn, batch_size=batch_size)
            train_loader = construct_train_loader(gold_set = bb_dataset_gold, collected_errors=collected_errors, batch_size=batch_size, collate_fn=collate_fn)
            print(f"Epoch {epoch}: Collected {len(collected_errors)} error samples for training.")

        model.train()

        train_acc = 0.0
        train_acc_pt = 0.0
        train_loss = 0.0
        num_batches = 0

        for step, (x_batch, y_batch) in enumerate(train_loader):
            num_batches += 1
            optimizer.zero_grad()

            logits, loss = model(x_batch, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            wandb.log({
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
            })

            train_acc += compute_accuracy(logits, y_batch)
            train_acc_pt += compute_accuracy_pt(logits, y_batch)
            train_loss += loss.item()
        
        print(f"Epoch {epoch}: Processed {num_batches} batches.")

        train_acc /= len(train_loader)
        train_acc_pt /= len(train_loader)
        train_loss /= len(train_loader)

        #   Compute performance on evaluation set

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
        #print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Train Acc PT: {train_acc_pt:.4f}, Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}, Test Acc PT: {test_acc_pt:.4f}, Test Loss: {test_loss:.4f}")
        wandb.log({
            "epoch": epoch,
            "train_acc": train_acc,
            "train_acc_pt": train_acc_pt,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "test_acc_pt": test_acc_pt,
            "test_loss": test_loss,
        })

    wandb.finish()

    print("Training finished! Saving model ...")

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
            "gold_size": gold_size,
            "error_pool_size": error_pool_size,
            "errors_per_epoch": errors_per_epoch,
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
        operation= Addition(),
    )

    train(
        name=args.name,
        model_name=args.model_name,
        digits=args.digits,
        train_size=args.train_size,
        test_size=args.test_size,
        eval_size=args.eval_size,
        error_correction=args.error_correction,
        gold_size=args.gold_size,
        error_pool_size=args.error_pool_size,
        errors_per_epoch=args.errors_per_epoch,
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
    parser.add_argument("--name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--digits", type=int)
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--eval_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--bb_height", type=int)
    parser.add_argument("--bb_width", type=int)
    parser.add_argument("--bb_randomize_position", action="store_true")
    parser.add_argument("--model_dimension", type=int)
    parser.add_argument("--num_heads_encoder", type=int)
    parser.add_argument("--n_encoder_blocks", type=int)
    parser.add_argument("--rope_mode", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gold_size", type=int)
    parser.add_argument("--error_pool_size", type=int)
    parser.add_argument("--errors_per_epoch", type=int)
    parser.add_argument("--error_correction", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    args, _ = parser.parse_known_args()
    main(args)
