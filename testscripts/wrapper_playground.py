# ------------------------------------------------------------
# wrapper_playground.py
#
# Convenient script to play around and test the blackboard reasoning model.
# (Marco) I might later replace this with a test suite, but for now, I do not have the time to write one.
# ------------------------------------------------------------
import torch
import logging

from src.models.eogar import EOgar
from src.evaluation.bb_chain_wrapper import BBChainReasoner, chainlist_to_results, BBChain
from projectlib.my_datasets.base import Split
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec, BlackboardSpec, Addition
from projectlib.my_datasets.collators import collate_bb_state_state, make_collator_with_args

from torch.optim import AdamW
from torch.utils.data import DataLoader

MODEL_NAME = "model_new.pth"

def check(train = False):

    spec = GenerationSpec(
        train_size = 5000,
        test_size = 5000,
        eval_size = 5000,
        low = 200,
        high = 400
    )

    bb_spec = BlackboardSpec(5, 10, False, Addition())

    bb_dataset = TokenizedBlackboardDataset(regenerate=True, generation_spec=spec, blackboard_spec=bb_spec, split=Split.TRAIN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 64
    vocab_size = bb_dataset.bb_2D_tokenizer.vocab_size
    d_model = 64

    collate_fn = make_collator_with_args(collate_bb_state_state, pad_token_id=bb_dataset.bb_2D_tokenizer.pad_id, device=device)
    data_loader = DataLoader(bb_dataset, batch_size=B, shuffle=False, collate_fn=collate_fn)

    print("Data loaded")

    model = EOgar(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads_encoder=8,
        n_encoder_blocks=4,
        pad_id=bb_dataset.bb_2D_tokenizer.pad_id
    ).to(device)

    print("Model initialized")

    if(train):
        optimizer = AdamW(model.parameters(), lr=0.5e-3)
        model.train()
        for epoch in range(2):
            for step, (x_batch, y_batch) in enumerate(data_loader):
                optimizer.zero_grad()

                logits, loss = model(x_batch, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        torch.save(model.state_dict(), MODEL_NAME)
    else:
        model.load_state_dict(torch.load(MODEL_NAME))

    reasoner = BBChainReasoner(model, torch.device(device), bb_spec, timeout_iters=8)

    print("Model loaded")
    st: BBChain = reasoner.compute_from_operands(10, 10) # TODO: fix generation here!
    st.show_steps()

    st = reasoner.compute_from_operands(20, 30)
    st.show_steps()
    print("Result is: ", st.result)


    st = reasoner.compute_from_operands(5, 19)
    st.show_steps()
    print("Result is: ", st.result)

    st = reasoner.compute_from_operands(241, 389)
    st.show_steps()
    print("Result is: ", st.result)
    

    st = reasoner.compute_from_operands(150, 280)
    st.show_steps()

    st = reasoner.compute_from_operands(909, 256)
    st.show_steps()

    # for i, [x, _] in enumerate(data_loader):
    #     chainlist = reasoner.compute_from_databatch(x)
    #     print(chainlist_to_results(chainlist))

        # for chain in chainlist:
        #     chain.show_steps()

if __name__ == "__main__":

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    check(True)
