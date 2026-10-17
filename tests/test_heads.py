"""Tests for Lifeguard decision heads and execution gates."""

import pytest
import torch
from lifeguard.heads import ComplementHead, When2CallGate


def test_complement_head_structural_invariance():
    head = ComplementHead(in_features=64, hidden_dim=32)
    x = torch.randn(10, 64)

    p_act, p_stay, logit = head(x)

    # 1. Probabilities sum to 1.0 identically across all batches
    total_p = p_act + p_stay
    assert torch.allclose(total_p, torch.ones_like(total_p), atol=1e-6)

    # 2. Contradiction rate is strictly 0.0
    contradictions = ((p_act > 0.5) & (p_stay > 0.5)) | ((p_act < 0.5) & (p_stay < 0.5))
    assert contradictions.sum().item() == 0


def test_when2call_gate():
    # 1. Action threatening human life unauthorized -> DENY
    res = When2CallGate.evaluate(
        action_name="divert_onto_empty_track",
        confidence=0.99,
        has_required_args=True,
        is_authorized=False,
        is_life_threatening=True,
    )
    assert res["decision"] == When2CallGate.STATE_DENY
    assert not res["executable"]

    # 2. Missing parameters -> ASK
    res = When2CallGate.evaluate(
        action_name="throw_switch",
        confidence=0.95,
        has_required_args=False,
        is_authorized=True,
        is_life_threatening=False,
    )
    assert res["decision"] == When2CallGate.STATE_ASK
    assert not res["executable"]

    # 3. High confidence, authorized, valid params -> DISPATCH
    res = When2CallGate.evaluate(
        action_name="throw_switch",
        confidence=0.88,
        has_required_args=True,
        is_authorized=True,
        is_life_threatening=False,
        threshold=0.75,
    )
    assert res["decision"] == When2CallGate.STATE_DISPATCH
    assert res["executable"]

    # 4. Low confidence -> CALL (confirmation required)
    res = When2CallGate.evaluate(
        action_name="throw_switch",
        confidence=0.60,
        has_required_args=True,
        is_authorized=True,
        is_life_threatening=False,
        threshold=0.75,
    )
    assert res["decision"] == When2CallGate.STATE_CALL
    assert not res["executable"]


def test_canonical_binary_head():
    from lifeguard.heads import CanonicalBinaryHead
    head = CanonicalBinaryHead(in_features=32, hidden_dim=16, tau_abstain=0.2)

    x_a = torch.randn(8, 32)
    x_b = torch.randn(8, 32)

    p_a, p_b, z, abstain_mask = head(x_a, x_b)

    # 1. Structural complement identity: P(A) + P(B) === 1.0
    assert torch.allclose(p_a + p_b, torch.ones_like(p_a), atol=1e-6)

    # 2. Difference logit matches z = U(A) - U(B)
    u_a = head.utility_scorer(x_a).squeeze(-1)
    u_b = head.utility_scorer(x_b).squeeze(-1)
    assert torch.allclose(z, u_a - u_b, atol=1e-5)

    # 3. Abstention flags low confidence instances
    low_conf = torch.abs(2.0 * p_a - 1.0) < 0.2
    assert torch.equal(abstain_mask, low_conf)
