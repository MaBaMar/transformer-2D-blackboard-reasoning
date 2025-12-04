# minimal training script for Edgar model
import torch
import logging

from src.models.edgar import Edgar
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec, BlackboardSpec, Addition, bb_datasample_prettyprint
from projectlib.my_datasets.collators import collate_blackboards, make_collator_with_args

from torch.optim import AdamW
from torch.utils.data import DataLoader

def check(train = False):

    spec = GenerationSpec(
        size = 1,
        low = 1,
        high = 5
    )

    bb_spec = BlackboardSpec(5, 5, False, Addition())

    bb_dataset = TokenizedBlackboardDataset(regenerate=True, generation_spec=spec, blackboard_spec=bb_spec)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 1
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
        pad_id=bb_dataset.bb_2D_tokenizer.pad_id,
        eos_id=bb_dataset.bb_2D_tokenizer.eos_id
    ).to(device)

    print("Model initialized")

    if(train):
        optimizer = AdamW(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(1):
            for step, (x_batch, y_batch) in enumerate(data_loader):
                # print("BOS ID: ",bb_dataset.bb_2D_tokenizer.bos_id)
                # print(y_batch[0])
                # exit(-1)
                optimizer.zero_grad()

                logits, loss = model(x_batch, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        torch.save(model.state_dict(), "model.pth")
    else:
        model.load_state_dict(torch.load("model.pth"))

    print("Model loaded")


if __name__ == "__main__":

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    check(True)
