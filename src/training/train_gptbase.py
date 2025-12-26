# ------------------------------------------------------------
# train_gptbase.py
#
# training script for GPT-Base model
# I tried to keep it similar to train_eogar.py, but added lr scheduling
# ------------------------------------------------------------

from argparse import ArgumentParser
import torch
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
from logging import getLogger, basicConfig, DEBUG
from tqdm import tqdm

from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup
import wandb
import numpy as np

from src.models.gptbase import GPTBaseTokenizer, GPTStyleBaseline, _DATA_T_REGISTRY, dataset_option_t
from src.evaluation.gptbase_wrapper import GPTBaseInferenceBatch
from projectlib.my_datasets import  GenerationSpec, Split
from projectlib.trainutils import compute_accuracy_pt

def compute_gptbase_digit_acc_from_stream(logits: torch.Tensor, labels: torch.Tensor, tokenizer: GPTBaseTokenizer, attn_mask: torch.Tensor) -> float:
    """
    Compute the accuracy of the GPTBase model's predictions based on the model output.
    We mask out all padding tokens at the end based on the attention mask. This ensures that garbage digits produced from padding tokens are ignored (same as in the loss).

    Args:
        logits (torch.Tensor): The logits output by the model.
        labels (torch.Tensor): The true labels (tensor of ints).
        tokenizer (GPTBaseTokenizer): The tokenizer used to encode the input.
        attn_mask (torch.Tensor): The attention mask.

    Returns:
        float: The accuracy of the model's predictions on this batch. The caller is responsible for averaging over batches and handling potentially different batch sizes.
    """
    preds: torch.Tensor = logits.argmax(dim=-1)
    assert preds.shape == attn_mask.shape, f"Predictions shape {preds.shape} does not match mask shape {attn_mask.shape}"

    # we want to ignore what the model outputs for the padding at the end
    preds[~attn_mask.bool()] = tokenizer._tok_internal.pad_token_id

    pred_dec = tokenizer.strip_decode(preds)
    b1 = GPTBaseInferenceBatch(pred_dec)

    token_wise = (b1.results == labels).float().mean().item()

    return token_wise

MODELS_PATH = "./models/"

def train(
    name: str,
    model_name: str,
    dataset_variant: dataset_option_t,
    train_size: int,
    test_size: int,
    eval_size: int,
    digits: int,
    batch_size: int,
    max_context_length: int,
    max_output_length: int,
    model_dimension: int,
    num_heads: int,
    n_decoder_blocks: int,
    learning_rate: float,
    epochs: int,
    seed: int,
    use_lr_scheduler: bool,
    warmup_steps: int,
    num_sched_cycles: float,
    link_weights: bool,
    logging: str = "local",
):
    if use_lr_scheduler:
        schedule_info = {
                "name": "cosine_annealing_with_warmup",
                "warmup_steps": warmup_steps,
                "num_cycles": num_sched_cycles
            }
    else:
        schedule_info = {
            "name": "constant_scheduler",
            "learning_rate": learning_rate
        }

    wandb.init(
        name=name,
        entity="blackboard-reasoning",
        project="blackboard-reasoning_gpt_baseline", # TODO change?
        config={
            "model": model_name,
            "dataset_variant": dataset_variant,
            "train_size": train_size,
            "test_size": test_size,
            "eval_size": eval_size,
            "digits": digits,
            "batch_size": batch_size,
            "max_context_length": max_context_length,
            "max_output_length": max_output_length,
            "model_dimension": model_dimension,
            "num_heads": num_heads,
            "n_decoder_blocks": n_decoder_blocks,
            "learning_rate": learning_rate,
            "scheduler": {
                **schedule_info
            },
            "link_weights": link_weights,
            "epochs": epochs,
            "seed": seed,
        },
        mode="online" if logging == "wandb" else "offline"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainlogger = getLogger(__name__ + os.path.basename(__file__))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = GenerationSpec.digits(
        digits=digits,
        eval_size=eval_size,
        test_size=test_size,
        train_size=train_size,
    )

    dataset_train = _DATA_T_REGISTRY[dataset_variant](
        regenerate=True,
        split=Split.TRAIN,
        seed=seed,
        generation_spec=spec,
    )

    dataset_test = _DATA_T_REGISTRY[dataset_variant](
        regenerate=True,
        split=Split.TEST,
        seed=seed,
        generation_spec=spec,
    )

    tokenizer: GPTBaseTokenizer = GPTBaseTokenizer(torch.device(device))

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    trainlogger.info("Data loaded.")

    model: GPTStyleBaseline = GPTStyleBaseline(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=max_context_length,
        d_model=model_dimension,
        num_heads=num_heads,
        num_blocks=n_decoder_blocks,
        token_config=tokenizer.get_token_config(),
        max_inference_steps=max_output_length,
        use_weight_linking=link_weights,
        # could add dropout, embedding dropout here as parameters, default is 0.1 for both
    ).to(device)

    trainlogger.info(f"Using model:\n{model}\nwith {sum(p.numel() for p in model.parameters())} parameters of which {2*sum(p.numel() for p in model.tok_emb.parameters())} are in the token embedding layer and logit generation layer in the final head.")

    num_steps = len(dataset_train) * epochs // batch_size
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    if use_lr_scheduler:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps, num_cycles=num_sched_cycles)
    else:
        scheduler = get_constant_schedule(optimizer)

    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        train_res_acc = 0.0
        train_pt_acc = 0.0
        train_loss = 0.0

        for step, data in enumerate(progress_bar):
            optimizer.zero_grad()

            tokenizer_out: dict[str, torch.Tensor] = tokenizer.encode_batch(data['input'], inference_mode=False)
            x = tokenizer_out['input_ids'][..., :-1]
            mask_raw = tokenizer_out['attention_mask']
            attention_mask = mask_raw[..., :-1]
            pred_mask = mask_raw[..., 1:] # for accuracy computation, we want to ignore all padding tokens at the end
            y = tokenizer_out["input_ids"][..., 1:]

            logits, loss = model(x, attention_mask, y)

            loss.backward()
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Update the progress bar with current metrics
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.6f}"
            })

            wandb.log({
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
                "lr": current_lr
            })

            train_res_acc += compute_gptbase_digit_acc_from_stream(logits, data['label'], tokenizer, pred_mask)
            train_pt_acc += compute_accuracy_pt(logits, y, ignore_tokens=tokenizer._tok_internal.pad_token_id)
            train_loss += loss.item()

        # Compute average metrics for the epoch
        train_res_acc /= len(train_loader)
        train_pt_acc /= len(train_loader)
        train_loss /= len(train_loader)

        # compute test set performance
        test_res_acc, test_pt_acc, test_loss = 0.0, 0.0, 0.0

        with torch.no_grad():
            model.eval()
            for data in test_loader:
                tokenizer_out = tokenizer.encode_batch(data['input'], inference_mode=False)
                x = tokenizer_out['input_ids'][..., :-1]
                mask_raw = tokenizer_out['attention_mask']
                attention_mask = mask_raw[..., :-1]
                pred_mask = mask_raw[..., 1:] # for accuracy computation, we want to ignore all padding tokens at the end
                y = tokenizer_out['input_ids'][..., 1:]
                logits, loss = model(x, attention_mask, y)
                test_res_acc += compute_gptbase_digit_acc_from_stream(logits, data['label'], tokenizer, pred_mask)
                test_pt_acc += compute_accuracy_pt(logits, y, ignore_tokens=tokenizer._tok_internal.pad_token_id)
                test_loss += loss.item()

        model.train()
        test_res_acc /= len(test_loader)
        test_pt_acc /= len(test_loader)
        test_loss /= len(test_loader)

        # Log epoch metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train_digit_acc": train_res_acc,
            "train_acc_pt": train_pt_acc,
            "train_loss": train_loss,
            "test_digit_acc": test_res_acc,
            "test_acc_pt": test_pt_acc,
            "test_loss": test_loss,
        })

    wandb.finish()

    trainlogger.info("Training finished! Saving model ...")

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    save_path = os.path.join(MODELS_PATH, f"{model_name}_d{digits}_s{seed}.pt")

    model.save_to_path(
        save_path,
        seed,
        warmup_steps=warmup_steps,
        epochs=epochs,
        train_size=train_size,
        digits=digits
    )

    trainlogger.info(f"Model saved to {save_path}.")


if __name__ == "__main__":
    # logging level
    basicConfig(level=DEBUG)

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-base")
    parser.add_argument("--dataset_variant", type=str, required=True, choices=_DATA_T_REGISTRY.keys())
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--test_size", type=int, required=True)
    parser.add_argument("--eval_size", type=int, required=True)
    parser.add_argument("--digits", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_context_length", type=int, required=True)
    parser.add_argument("--max_output_length", type=int, required=True)
    parser.add_argument("--model_dimension", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--n_decoder_blocks", type=int, required=True)
    parser.add_argument("--link_weights", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging", type=str, default="local")
    parser.add_argument("--use_lr_scheduler", type=bool, default=False)
    parser.add_argument("--warmup_steps", type=int, default=10)         # number of warmup steps for the learning rate scheduler, only used if --use_lr_scheduler is True
    parser.add_argument("--num_sched_cycles", type=float, default=0.5)    # number of cycles for the learning rate scheduler, only used if --use_lr_scheduler is True

    args = parser.parse_args()
    train(**vars(args))
