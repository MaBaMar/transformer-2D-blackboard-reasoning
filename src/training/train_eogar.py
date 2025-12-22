# minimal training script for EOgar model
import argparse
import os
import torch
import logging

import wandb
import numpy as np

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.eogar import EOgar
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec, Split, BlackboardSpec, Addition
from projectlib.my_datasets.collators import collate_bb_state_state, make_collator_with_args
from projectlib.trainutils import compute_accuracy_pt, compute_accuracy


MODELS_PATH = "./models/"

def train(
        name: str,
        model_name: str,
        train_size: int,
        test_size: int,
        eval_size: int,
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

    bb_dataset_train = TokenizedBlackboardDataset(
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

    pad_id = bb_dataset_train.bb_2D_tokenizer.pad_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = bb_dataset_train.bb_2D_tokenizer.vocab_size

    collate_fn = make_collator_with_args(collate_bb_state_state, pad_token_id=pad_id, device=device)
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

    print("Data loaded.")

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

        #   Train the model on the training set

        model.train()

        train_acc = 0.0
        train_acc_pt = 0.0
        train_loss = 0.0

        for step, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            y_batch = y_batch[0] # we only care about the first entry in the tuple

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
                y_batch = y_batch.to(device)
                logits, loss = model(x_batch, y_batch)

                test_acc += compute_accuracy(logits, y_batch)
                test_acc_pt += compute_accuracy_pt(logits, y_batch)
                test_loss += loss.item()

        test_acc /= len(test_loader)
        test_acc_pt /= len(test_loader)
        test_loss /= len(test_loader)

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
    parser.add_argument("--bb_randomize_position", type=bool)
    parser.add_argument("--model_dimension", type=int)
    parser.add_argument("--num_heads_encoder", type=int)
    parser.add_argument("--n_encoder_blocks", type=int)
    parser.add_argument("--rope_mode", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    args, _ = parser.parse_known_args()
    main(args)
