"""Tests for Lifeguard loss functions."""

import pytest
import torch
from lifeguard.loss import TripartiteLifeguardLoss, UnanchoredContrastiveLoss


def test_tripartite_loss_computation():
    loss_fn = TripartiteLifeguardLoss(lambda_comp=1.0, lambda_mono=0.5, margin=0.1)

    # Perfectly aligned predictions: P(act) matches targets, P(stay) = 1 - P(act)
    p_act = torch.tensor([0.95, 0.05, 0.90, 0.10], requires_grad=True)
    p_stay = torch.tensor([0.05, 0.95, 0.10, 0.90], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    stakes_diff = torch.tensor([5.0, -2.0, 3.0, -1.0])

    loss, metrics = loss_fn(p_act, p_stay, targets, stakes_diff)
    assert loss.item() > 0.0
    assert metrics["loss_comp"] < 1e-4  # Perfectly complementary
    assert metrics["contradiction_rate"] == 0.0

    # Backpropagation succeeds
    loss.backward()
    assert p_act.grad is not None
    assert p_stay.grad is not None


def test_contradiction_penalization():
    loss_fn = TripartiteLifeguardLoss(lambda_comp=2.0)

    # Contradictory predictions: model says "yes" to both actions (P > 0.5)
    p_act = torch.tensor([0.9, 0.8], requires_grad=True)
    p_stay = torch.tensor([0.9, 0.8], requires_grad=True)
    targets = torch.tensor([1.0, 1.0])

    loss, metrics = loss_fn(p_act, p_stay, targets)
    assert metrics["contradiction_rate"] == 1.0
    assert metrics["mean_polarity_gap"] > 0.5
    assert metrics["loss_comp"] > 0.3


def test_unanchored_contrastive_collapse():
    """Demonstrate why naive L_contrast = (s1 + s2)^2 trivially collapses to s -> 0."""
    loss_fn = UnanchoredContrastiveLoss()

    # Logits initialized away from 0
    s_act = torch.tensor([2.5, -1.8, 3.0], requires_grad=True)
    s_stay = torch.tensor([2.5, -1.8, 3.0], requires_grad=True)

    optimizer = torch.optim.SGD([s_act, s_stay], lr=0.1)

    # Optimize solely with unanchored contrastive loss
    for _ in range(50):
        optimizer.zero_grad()
        loss = loss_fn(s_act, s_stay)
        loss.backward()
        optimizer.step()

    # Sum of logits approaches 0
    assert torch.allclose(s_act + s_stay, torch.zeros_like(s_act), atol=1e-2)
