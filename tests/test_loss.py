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


def test_negative_utility_penalization():
    """Verify that negative utility difference (Delta U < 0) penalizes preferring action."""
    loss_fn = TripartiteLifeguardLoss(lambda_comp=0.0, lambda_mono=1.0, margin=0.2, tau_scale=5.0)

    # Action A is worse by 5 units: Delta U = -5.0
    stakes_diff = torch.tensor([-5.0])
    targets = torch.tensor([0.0])

    # Case 1: Model errantly prefers Action A (p_act = 0.8, p_stay = 0.2)
    p_act_wrong = torch.tensor([0.8])
    p_stay_wrong = torch.tensor([0.2])
    _, metrics_wrong = loss_fn(p_act_wrong, p_stay_wrong, targets, stakes_diff)

    # Case 2: Model correctly prefers Action B / stay (p_act = 0.1, p_stay = 0.9)
    p_act_right = torch.tensor([0.1])
    p_stay_right = torch.tensor([0.9])
    _, metrics_right = loss_fn(p_act_right, p_stay_right, targets, stakes_diff)

    # The wrong choice must incur substantial mono loss, while correct choice incurs zero mono loss
    assert metrics_wrong["loss_mono"] > 0.7  # m + z = 0.2*tanh(1) + 0.6 ~ 0.752
    assert metrics_right["loss_mono"] == 0.0


def test_tie_neutrality_penalization():
    """Verify that ties (Delta U == 0) penalize non-indifference via quadratic penalty lambda * z^2."""
    loss_fn = TripartiteLifeguardLoss(lambda_comp=0.0, lambda_mono=1.0, lambda_tie=1.0)

    # Tie situation: Delta U = 0
    stakes_diff = torch.tensor([0.0])
    targets = torch.tensor([0.5])

    # Biased prediction on a tie: p_act = 0.8, p_stay = 0.2 -> z = 0.6
    p_act_biased = torch.tensor([0.8])
    p_stay_biased = torch.tensor([0.2])
    _, metrics_biased = loss_fn(p_act_biased, p_stay_biased, targets, stakes_diff)

    # Indifferent prediction on a tie: p_act = 0.5, p_stay = 0.5 -> z = 0.0
    p_act_neutral = torch.tensor([0.5])
    p_stay_neutral = torch.tensor([0.5])
    _, metrics_neutral = loss_fn(p_act_neutral, p_stay_neutral, targets, stakes_diff)

    assert pytest.approx(metrics_biased["loss_mono"], rel=1e-3) == 0.36  # (0.6)^2
    assert metrics_neutral["loss_mono"] == 0.0
