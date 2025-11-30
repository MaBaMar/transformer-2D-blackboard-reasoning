# minimal training script for Edgar model
import argparse
import torch
import wandb
import numpy as np

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.edgar import Edgar
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec
from projectlib.my_datasets.collators import collate_blackboards, make_collator_with_args



def train(
        name: str,
        model_name: str,
        size: int,
        digits: int,
        batch_size: int,
        model_dimension: int,
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
            "size": size,
            "digits": digits,
            "batch_size": batch_size,
            "model_dimension": model_dimension,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "seed": seed,
        },
        mode="online" if logging == "wandb" else "offline"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = GenerationSpec(
        size = size,
        low = 10**(digits - 1),
        high = 10**(digits)
    )

    bb_dataset = TokenizedBlackboardDataset(regenerate=True, generation_spec=spec)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = bb_dataset.bb_2D_tokenizer.vocab_size

    collate_fn = make_collator_with_args(collate_blackboards, pad_token_id=bb_dataset.bb_2D_tokenizer.pad_id, device=device)
    data_loader = DataLoader(bb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    print("Data loaded")

    model = Edgar(
        vocab_size=vocab_size,
        d_model=model_dimension,
        num_heads_encoder=4,
        num_heads_decoder=4,
        n_encoder_blocks=2,
        n_decoder_blocks=2,
        pad_id=bb_dataset.bb_2D_tokenizer.pad_id
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()

    print("Model initialized")

    for epoch in tqdm(range(epochs)):
        for step, (x_batch, y_batch) in enumerate(data_loader):
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

    wandb.finish()



def main(args):
    train(
        name=args.name,
        model_name=args.model_name,
        digits=args.digits,
        size=args.size,
        batch_size=args.batch_size,
        model_dimension=args.model_dimension,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        logging=args.logging,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--digits", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--model_dimension", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    args, _ = parser.parse_known_args()
    main(args)
