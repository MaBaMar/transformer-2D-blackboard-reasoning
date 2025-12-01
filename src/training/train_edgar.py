# minimal training script for Edgar model
import torch
import logging

from src.models.edgar import Edgar
from src.evaluation.bb_chain_wrapper import BBChain, BBChainReasoner
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec, BlackboardSpec, Addition, bb_prettyprint
from projectlib.my_datasets.collators import collate_blackboards, make_collator_with_args

from torch.optim import AdamW
from torch.utils.data import DataLoader

def check(train = False):

    spec = GenerationSpec(
        size = 1,
        low = 80,
        high = 200
    )

    bb_spec = BlackboardSpec(8, 10, True, Addition())

    bb_dataset = TokenizedBlackboardDataset(regenerate=True, generation_spec=spec, blackboard_spec=bb_spec)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 64
    vocab_size = bb_dataset.bb_2D_tokenizer.vocab_size
    d_model = 64

    collate_fn = make_collator_with_args(collate_blackboards, pad_token_id=bb_dataset.bb_2D_tokenizer.pad_id, device=device)
    data_loader = DataLoader(bb_dataset, batch_size=B, shuffle=False, collate_fn=collate_fn)

    print("Data loaded")

    model = Edgar(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads_encoder=8,
        num_heads_decoder=8,
        n_encoder_blocks=4,
        n_decoder_blocks=4,
        pad_id=bb_dataset.bb_2D_tokenizer.pad_id,
        eos_id=bb_dataset.bb_2D_tokenizer.eos_id
    ).to(device)

    print("Model initialized")

    if(train):
        optimizer = AdamW(model.parameters(), lr=0.5e-3)
        model.train()
        for epoch in range(10):
            for step, (x_batch, y_batch) in enumerate(data_loader):
                optimizer.zero_grad()

                logits, loss = model(x_batch, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        torch.save(model.state_dict(), "model.pth")
    else:
        model.load_state_dict(torch.load("model.pth"))

    reasoner = BBChainReasoner(model, torch.device(device), bb_spec, timeout_iters=8)

    print("Model loaded")
    st: BBChain = reasoner.compute_from_operands(10, 10) # TODO: fix generation here!
    st.show_steps()

    st = reasoner.compute_from_operands(50, 80)
    st.show_steps()

    st = reasoner.compute_from_operands(150, 280)
    st.show_steps()

    st = reasoner.compute_from_operands(909, 256)
    st.show_steps()

    # for x, y in data_loader:
    #     print(model.forward(x, y)[0].argmax(dim=-1))

    #     print("train_prediction:")
    #     bb_prettyprint(model.forward(x, y)[0].argmax(dim=-1), bb_dataset.bb_2D_tokenizer)
    # print()

if __name__ == "__main__":

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    check(False)
