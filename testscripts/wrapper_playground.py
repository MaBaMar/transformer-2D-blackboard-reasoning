# ------------------------------------------------------------
# wrapper_playground.py
#
# Convenient script to play around and test the blackboard reasoning model.
# (Marco) I might later replace this with a test suite, but for now, I do not have the time to write one.
# ------------------------------------------------------------
import torch
import logging
from tqdm import tqdm

from src.models.eogar import EOgar
from src.evaluation.bb_chain_wrapper import BBChainReasoner, chainlist_to_results, BBChain
from projectlib.my_datasets.base import Split
from projectlib.my_datasets.blackboards import TokenizedBlackboardDataset, GenerationSpec, BlackboardSpec, Addition
from projectlib.my_datasets.collators import collate_bb_state_state, collate_bb_state_int, make_collator_with_args

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

MODEL_NAME = "eogar_save_test.pth"

def check(train = False):

    spec = GenerationSpec(
        train_size = 8192,
        test_size = 8192,
        eval_size = 1024,
        low = 1000,
        high = 10**12
    )

    bb_spec = BlackboardSpec(5, 20, False, Addition())

    bb_dataset = TokenizedBlackboardDataset(regenerate=False, generation_spec=spec, blackboard_spec=bb_spec, split=Split.TRAIN)
    # exit(-1)
    bb_dataset_eval = TokenizedBlackboardDataset(regenerate=False, generation_spec=spec, blackboard_spec=bb_spec, split=Split.EVAL)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 64
    vocab_size = bb_dataset.bb_2D_tokenizer.vocab_size
    d_model = 64

    collate_fn_train = make_collator_with_args(collate_bb_state_state, pad_token_id=bb_dataset.bb_2D_tokenizer.pad_id, device=device)
    data_loader_train = DataLoader(bb_dataset, batch_size=B, shuffle=True, collate_fn=collate_fn_train)

    collate_fn_eval = make_collator_with_args(collate_bb_state_int, pad_token_id=bb_dataset.bb_2D_tokenizer.pad_id, device=device)
    data_loader_eval = DataLoader(bb_dataset_eval, batch_size=B, shuffle=False, collate_fn=collate_fn_eval)

    print("Data loaded")

    model = EOgar(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads_encoder=8,
        n_encoder_blocks=12,
        pad_id=bb_dataset.bb_2D_tokenizer.pad_id
    ).to(device)

    # model.load_state_dict(torch.load("model_new.pth"))
    print("Model initialized")

    if(train):
        epochs = 5
        optimizer = AdamW(model.parameters(), lr=1e-3)
        total_steps = len(data_loader_train) * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

        model.train()

        for epoch in range(epochs):
            progress_bar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

            for step, (x_batch, y_batch) in enumerate(progress_bar):
                # Optional: Move data to device if not done inside loader
                # x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()

                logits, loss = model(x_batch, y_batch)

                loss.backward()
                optimizer.step()

                scheduler.step()

                current_lr = scheduler.get_last_lr()[0]

                # Update the progress bar with current metrics
                # This replaces the print statement
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.6f}"
                })
            if epoch % 5 == 0:
                torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
                acc = 0
                reasoner = BBChainReasoner(model, torch.device(device), bb_spec, timeout_iters=15)
                for i, [x, y] in enumerate(data_loader_eval):
                    chainlist = reasoner.compute_from_databatch(x)
                    # print(chainlist_to_results(chainlist))
                    acc += (chainlist_to_results(chainlist).to(device) == y).sum().item()
                acc /= len(data_loader_eval)

                print(f"Accuracy: {acc:.4f}")

        torch.save(model.state_dict(), MODEL_NAME)
    else:
        model.load_state_dict(torch.load(MODEL_NAME))

    reasoner = BBChainReasoner(model, torch.device(device), bb_spec, timeout_iters=15)

    acc = 0
    cnt = 0
    for i, [x, y] in enumerate(data_loader_eval):
        chainlist = reasoner.compute_from_databatch(x)
        # print(chainlist_to_results(chainlist))
        acc += (chainlist_to_results(chainlist).to(device) == y).sum().item()
        cnt += x[0].shape[0]
    acc /= cnt

    print(f"Accuracy: {acc:.4f}")

    print("Model loaded")
    st: BBChain = reasoner.compute_from_operands(67544657, 12342564) # TODO: fix generation here!
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

    for i, [x, _] in enumerate(data_loader):
        chainlist = reasoner.compute_from_databatch(x)
        print(chainlist_to_results(chainlist))

        for chain in chainlist:
            chain.show_steps()

if __name__ == "__main__":

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    check(True)
