# minimal training script for Edgar model
import torch
from src.models.edgar import Edgar

from torch.optim import AdamW

def get_dummy_batch(batch_size, seq_len, vocab_size, device):
    """Generates random tensors matching the (tokens, row, col, mask) format."""

    # 1. Random Tokens
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

    # 2. Random Positions (0 to 9)
    pos_row = torch.randint(0, 10, (batch_size, seq_len), device=device)
    pos_col = torch.randint(0, 10, (batch_size, seq_len), device=device)

    # 3. Mask (False = Keep, True = Ignore)
    # Let's make the last few tokens padding to test masking logic
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    mask[:, -2:] = True # Mask out last 2 tokens
    tokens[:, -2:] = 0  # Set padded tokens to 0 (pad_id)

    return (tokens, pos_row, pos_col, mask)

def minimal_check_edgar():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, L_src, L_tgt = 2, 16, 16
    vocab_size = 100
    d_model = 64

    model = Edgar(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads_encoder=4,
        num_heads_decoder=4,
        n_encoder_blocks=2,
        n_decoder_blocks=2,
        pad_id=0,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    x_batch = get_dummy_batch(B, L_src, vocab_size, device)
    y_batch = get_dummy_batch(B, L_tgt, vocab_size, device)

    model.train()
    for step in range(5):
        optimizer.zero_grad()

        # Forward pass
        logits, loss = model(x_batch, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            generated_ids = model.next_state(x_batch)

        print(f"Step {step+1}, Loss: {loss.item():.4f}, Logit Shape: {logits.shape}, IDs Shape: {generated_ids.shape}")

if __name__ == "__main__":
    minimal_check_edgar()
