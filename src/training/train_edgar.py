# minimal training script for Edgar model
import torch
from src.models.edgar import Edgar
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec
from projectlib.my_datasets.collators import collate_blackboards, make_collator_with_args

from torch.optim import AdamW
from torch.utils.data import DataLoader

def minimal_check_edgar():

    spec = GenerationSpec(
        size = 1000,
        low = 10,
        high = 1000
    )

    bb_dataset = TokenizedBlackboardDataset(regenerate=True, generation_spec=spec)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, L_src, L_tgt = 32, 16, 16
    vocab_size = bb_dataset.bb_2D_tokenizer.vocab_size
    d_model = 64

    collate_fn = make_collator_with_args(collate_blackboards, pad_token_id=bb_dataset.bb_2D_tokenizer.pad_id, device=device)
    data_loader = DataLoader(bb_dataset, batch_size=B, shuffle=True, collate_fn=collate_fn)

    print("Data loaded")

    model = Edgar(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads_encoder=4,
        num_heads_decoder=4,
        n_encoder_blocks=2,
        n_decoder_blocks=2,
        pad_id=bb_dataset.bb_2D_tokenizer.pad_id
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    model.train()

    print("Model initialized")

    for epoch in range(3):
        for step, (x_batch, y_batch) in enumerate(data_loader):
            optimizer.zero_grad()

            logits, loss = model(x_batch, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")


if __name__ == "__main__":
    minimal_check_edgar()
