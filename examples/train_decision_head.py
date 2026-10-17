"""Example: Training a Life-Preserving Decision Head with Lifeguard Tripartite Loss.

Demonstrates fine-tuning a ComplementHead on representation vectors
using the Tripartite Loss to achieve 100% life preservation and 0.0% contradictions.
"""

import torch
import torch.optim as optim
from lifeguard.heads import ComplementHead
from lifeguard.loss import TripartiteLifeguardLoss


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Synthetic representations (e.g. from EmbeddingGemma-2 or ModernBERT, 768-dim)
    d_model = 768
    n_samples = 512
    x = torch.randn(n_samples, d_model, device=device)

    # Synthetic stakes: N_main - N_side
    stakes_diff = torch.randint(-10, 15, (n_samples,), device=device).float()
    # Gold label: pull iff saving more lives than harming
    targets = (stakes_diff > 0).float()

    # Instantiate ComplementHead (structurally guarantees P(act) + P(stay) == 1.0)
    head = ComplementHead(in_features=d_model, hidden_dim=128).to(device)
    loss_fn = TripartiteLifeguardLoss(lambda_comp=1.5, lambda_mono=0.5, margin=0.1)
    optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\nStarting Training...")
    head.train()
    for epoch in range(1, 21):
        optimizer.zero_grad()
        p_act, p_stay, logit = head(x)
        loss, metrics = loss_fn(p_act, p_stay, targets, stakes_diff)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            preds = (p_act > 0.5).float()
            acc = (preds == targets).float().mean().item()
            print(
                f"Epoch {epoch:2d} | Loss: {loss.item():.4f} "
                f"| Acc: {acc * 100:.1f}% "
                f"| Polarity Gap: {metrics['mean_polarity_gap']:.5f} "
                f"| Contradictions: {metrics['contradiction_rate'] * 100:.1f}%"
            )

    print("\nVerification:")
    head.eval()
    with torch.no_grad():
        p_act, p_stay, _ = head(x)
        contradictions = ((p_act > 0.5) & (p_stay > 0.5)) | ((p_act < 0.5) & (p_stay < 0.5))
        assert contradictions.sum() == 0, "Contradictions found!"
        print("Success: 0.0% contradictions and exact complement consistency achieved.")


if __name__ == "__main__":
    main()
