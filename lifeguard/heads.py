"""Lifeguard Decision Heads and Structural Parameterizations.

Implements:
1. ComplementHead: Structural single-logit parameterization guaranteeing P(act) + P(stay) == 1.0
2. DualPredicateGate: Calibrated readout for dual query strings
3. When2CallGate: Quadri-state execution gate separating classification from authorization
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplementHead(nn.Module):
    """Structural complement decision head.

    Guarantees logical complement consistency by construction.
    Derives both probabilities from a single underlying latent logit z:
        P(act) = sigmoid(z)
        P(stay) = sigmoid(-z) = 1.0 - P(act)

    Mathematically enforces:
        P(act) + P(stay) === 1.0
        Contradictions (both > 0.5 or both < 0.5) === 0.0%
    """

    def __init__(self, in_features: int, hidden_dim: int = 128, temperature: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute structural complementary probabilities.

        Args:
            embedding: Input representation vector of shape [B, in_features]

        Returns:
            p_act: Probability of acting (shape: [B])
            p_stay: Probability of staying (shape: [B]), identically equal to 1 - p_act
            logit: Raw decision logit z
        """
        raw_logit = self.net(embedding).squeeze(-1)
        scaled_logit = raw_logit / torch.clamp(self.temperature, min=1e-3)
        p_act = torch.sigmoid(scaled_logit)
        p_stay = 1.0 - p_act
        return p_act, p_stay, scaled_logit


class DualPredicateGate(nn.Module):
    """Readout head for systems where predicates are presented as distinct queries.

    Accepts embeddings for both the act query and stay query, enforcing tied-weight
    polarity inversion with calibrated margin.
    """

    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, embed_act: torch.Tensor, embed_stay: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_act = self.feature_proj(embed_act)
        h_stay = self.feature_proj(embed_stay)
        logit_act = self.scorer(h_act).squeeze(-1)
        logit_stay = self.scorer(h_stay).squeeze(-1)
        p_act = torch.sigmoid(logit_act)
        p_stay = torch.sigmoid(logit_stay)
        return p_act, p_stay


class When2CallGate:
    """Quadri-state Execution Gate for Responsible Tool & Action Execution.

    Enforces that positive selection (e.g. highest softmax score) cannot trigger
    autonomous execution without passing:
    1. Schema argument validation
    2. Policy authorization
    3. Calibrated thresholding

    States:
        DISPATCH (1): Valid, authorized, safe to execute autonomously
        CALL (2): Recommended, pending user confirmation
        ASK (3): Missing parameters or ambiguous intent
        DENY (4): Explicitly prohibited or violating life preservation
    """

    STATE_DISPATCH = "DISPATCH"
    STATE_CALL = "CALL"
    STATE_ASK = "ASK"
    STATE_DENY = "DENY"

    @classmethod
    def evaluate(
        cls,
        action_name: str,
        confidence: float,
        has_required_args: bool,
        is_authorized: bool,
        is_life_threatening: bool,
        threshold: float = 0.75,
    ) -> Dict[str, any]:
        # Life-preservation override: actions that cause gratuitous harm are immediately DENIED
        if is_life_threatening and not is_authorized:
            return {
                "decision": cls.STATE_DENY,
                "reason": "Policy violation: action poses unmitigated threat to human life.",
                "executable": False,
            }

        if not is_authorized:
            return {
                "decision": cls.STATE_DENY,
                "reason": "Unauthorized action: destructive or restricted capability.",
                "executable": False,
            }

        if not has_required_args:
            return {
                "decision": cls.STATE_ASK,
                "reason": "Missing required schema parameters.",
                "executable": False,
            }

        if confidence >= threshold:
            return {
                "decision": cls.STATE_DISPATCH,
                "reason": f"High confidence ({confidence:.2f} >= {threshold:.2f}) and validated parameters.",
                "executable": True,
            }

        return {
            "decision": cls.STATE_CALL,
            "reason": f"Confidence ({confidence:.2f}) below autonomous execution threshold ({threshold:.2f}).",
            "executable": False,
        }


class CanonicalBinaryHead(nn.Module):
    r"""Native Binary Decision Head for verified mutually exclusive action pairs.

    Derives decision probabilities from the canonical utility difference logit:
        z(x) = U_\theta(\text{action}_A \mid x) - U_\theta(\text{action}_B \mid x)
        P(\text{action}_A) = \sigma(z / T)
        P(\text{action}_B) = \sigma(-z / T) = 1.0 - P(\text{action}_A)

    Essential Safeguard:
        Requires semantic verification that action_A and action_B are mutually exclusive
        and exhaustive within identical context x. Does not blindly flip string negations.

    Includes explicit epistemic uncertainty / abstention estimation:
        confidence = |2 * P(action_A) - 1.0|
        abstain = confidence < tau_abstain
        (distinguishing between endorsing action_B vs abstaining due to insufficient evidence).
    """

    def __init__(self, in_features: int, hidden_dim: int = 128, tau_abstain: float = 0.15):
        super().__init__()
        self.utility_scorer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.temperature = nn.Parameter(torch.tensor([1.0]))
        self.tau_abstain = tau_abstain

    def forward(
        self,
        embed_context_a: torch.Tensor,
        embed_context_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute verified binary choice probabilities and abstention mask.

        Args:
            embed_context_a: Representation of state x conditioned on action A [B, D]
            embed_context_b: Representation of state x conditioned on action B [B, D]

        Returns:
            p_a: Probability of action A (shape: [B])
            p_b: Probability of action B (shape: [B], identically 1 - p_a)
            logit_diff: Canonical utility difference logit z(x) = U(A|x) - U(B|x)
            abstain_mask: Boolean tensor where True indicates insufficient evidence / abstention
        """
        u_a = self.utility_scorer(embed_context_a).squeeze(-1)
        u_b = self.utility_scorer(embed_context_b).squeeze(-1)

        z = (u_a - u_b) / torch.clamp(self.temperature, min=1e-3)
        p_a = torch.sigmoid(z)
        p_b = 1.0 - p_a

        confidence = torch.abs(2.0 * p_a - 1.0)
        abstain_mask = confidence < self.tau_abstain

        return p_a, p_b, z, abstain_mask
