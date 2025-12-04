# minimal training script for Edgar model
import argparse
import os
import torch
import logging

import wandb
import numpy as np

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.edgar import Edgar
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec, BlackboardSpec, Addition, bb_datasample_prettyprint
from projectlib.my_datasets.collators import collate_blackboards, make_collator_with_args



MODELS_PATH = "./models/"



def compute_accuracy(logits, labels, pad_id=None):
    labels = labels[0]
    preds = logits.argmax(dim=-1)
    
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)

    if labels.shape != preds.shape:
        return 0

    if pad_id is not None:
        mask = (labels != pad_id).float()
        correct = (preds == labels).float() * mask
        return correct.sum().item() / mask.sum().item()
    else:
        return (preds == labels).float().mean().item()



def train(
        name: str,
        model_name: str,
        train_size: int,
        eval_size: int,
        digits: int,
        batch_size: int,
        model_dimension: int,
        num_heads_encoder: int,
        num_heads_decoder: int,
        n_encoder_blocks: int,
        n_decoder_blocks: int,
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
            "eval_size": eval_size,
            "digits": digits,
            "batch_size": batch_size,
            "model_dimension": model_dimension,
            "num_heads_encoder": num_heads_encoder,
            "num_heads_decoder": num_heads_decoder,
            "n_encoder_blocks": n_encoder_blocks,
            "n_decoder_blocks": n_decoder_blocks,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "seed": seed,
        },
        mode="online" if logging == "wandb" else "offline"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    bb_dataset_train = TokenizedBlackboardDataset(
        regenerate=True,
        seed=seed,
        generation_spec=GenerationSpec.digits(
            size=train_size,
            digits=digits
        ),
    )

    bb_dataset_eval = TokenizedBlackboardDataset(
        regenerate=True,
        train=False,
        seed=seed,
        generation_spec=GenerationSpec.digits(
            size=eval_size,
            digits=digits
        ),
    )

    pad_id = bb_dataset_train.bb_2D_tokenizer.pad_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = bb_dataset_train.bb_2D_tokenizer.vocab_size

    collate_fn = make_collator_with_args(collate_blackboards, pad_token_id=pad_id, device=device)
    train_loader = DataLoader(
        bb_dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    eval_loader = DataLoader(
        bb_dataset_eval, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    print("Data loaded.")

    model = Edgar(
        vocab_size=vocab_size,
        d_model=model_dimension,
        num_heads_encoder=num_heads_encoder,
        num_heads_decoder=num_heads_decoder,
        n_encoder_blocks=n_encoder_blocks,
        n_decoder_blocks=n_decoder_blocks,
        pad_id=pad_id,
        eos_id=bb_dataset_train.bb_2D_tokenizer.eos_id,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params} parameters.")

    for epoch in tqdm(range(epochs)):

        #   Train the model on the training set

        model.train()

        train_acc = 0.0
        train_loss = 0.0

        for step, (x_batch, y_batch) in enumerate(train_loader):
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

            train_acc += compute_accuracy(logits, y_batch, pad_id)
            train_loss += loss.item()

        train_acc /= len(train_loader)
        train_loss /= len(train_loader)

        #   Compute performance on evaluation set

        model.eval()

        eval_acc = 0.0
        eval_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in eval_loader:
                logits, loss = model(x_batch, y_batch)

                eval_acc += compute_accuracy(logits, y_batch, pad_id)
                eval_loss += loss.item()

        eval_acc /= len(eval_loader)
        eval_loss /= len(eval_loader)

        wandb.log({
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "eval_acc": eval_acc,
            "eval_loss": eval_loss,
        })

    wandb.finish()

    print("Training finished! Saving model ...")

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    save_path = os.path.join(MODELS_PATH, f"{model_name}_e{epochs}_s{train_size}_d{digits}_md{model_dimension}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "d_model": model_dimension,
            "num_heads_encoder": 4,
            "num_heads_decoder": 4,
            "n_encoder_blocks": 2,
            "n_decoder_blocks": 2,
        }
    }, save_path)

    print(f"Model saved to {save_path}.")



def main(args):
    train(
        name=args.name,
        model_name=args.model_name,
        digits=args.digits,
        train_size=args.train_size,
        eval_size=args.eval_size,
        batch_size=args.batch_size,
        model_dimension=args.model_dimension,
        num_heads_encoder=args.num_heads_encoder,
        num_heads_decoder=args.num_heads_decoder,
        n_encoder_blocks=args.n_encoder_blocks,
        n_decoder_blocks=args.n_decoder_blocks,
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
    parser.add_argument("--eval_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--model_dimension", type=int)
    parser.add_argument("--num_heads_encoder", type=int)
    parser.add_argument("--num_heads_decoder", type=int)
    parser.add_argument("--n_encoder_blocks", type=int)
    parser.add_argument("--n_decoder_blocks", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    args, _ = parser.parse_known_args()
    main(args)
