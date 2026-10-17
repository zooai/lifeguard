"""Lifeguard Objective Functions & Tripartite Loss.

Provides mathematically sound loss functions for:
1. Supervised decision accuracy (L_sup)
2. Complement consistency between mutually exclusive predicates (L_comp)
3. Monotonic consequence scaling with stakes magnitude (L_mono)

Prevents trivial representation collapse associated with unanchored contrast penalties.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripartiteLifeguardLoss(nn.Module):
    r"""Tripartite Lifeguard Loss.

    Combines:
    1. Supervised task accuracy anchor:
       \mathcal{L}_{sup} = \text{BCE}(P(act), y)

    2. Polar complement consistency:
       \mathcal{L}_{comp} = (P(act) + P(stay) - 1.0)^2

       Note on unanchored quadratic penalties:
       A naive contrast penalty \mathcal{L}_{contrast} = (s(act) + s(stay))^2
       suffers from a degenerate global minimum at s(act) = s(stay) = 0.
       In logit space, this drives both probabilities to P = 0.5 (maximal entropy
       coin-flip), destroying predictive discrimination.
       The Lifeguard Tripartite loss fixes this by anchoring with \mathcal{L}_{sup}
       and operating directly on bounded probabilities: (P(act) + P(stay) - 1)^2.

    3. Monotonic consequence ranking:
       \mathcal{L}_{mono} = \max(0, \gamma - (s(N_{save}) - s(N_{harm})))
       Ensuring that as the net human lives saved increases, the decision margin
       monotonically strengthens.
    """

    def __init__(
        self,
        lambda_comp: float = 1.0,
        lambda_mono: float = 0.5,
        margin: float = 0.2,
    ):
        super().__init__()
        self.lambda_comp = lambda_comp
        self.lambda_mono = lambda_mono
        self.margin = margin
        self.bce = nn.BCELoss(reduction="mean")

    def forward(
        self,
        p_act: torch.Tensor,
        p_stay: torch.Tensor,
        targets: torch.Tensor,
        stakes_diff: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the tripartite loss.

        Args:
            p_act: Predicted probability of acting (shape: [B])
            p_stay: Predicted probability of abstaining / staying (shape: [B])
            targets: Binary ground truth labels for acting (shape: [B])
            stakes_diff: Difference in lives saved (N_save - N_harm) (shape: [B], optional)

        Returns:
            total_loss: Scalar composite loss
            metrics: Dictionary of component loss values
        """
        # Ensure 1D tensors
        p_act = p_act.view(-1)
        p_stay = p_stay.view(-1)
        targets = targets.view(-1).float()

        # 1. Supervised Anchor Loss
        loss_sup_act = self.bce(p_act, targets)
        loss_sup_stay = self.bce(p_stay, 1.0 - targets)
        loss_sup = 0.5 * (loss_sup_act + loss_sup_stay)

        # 2. Polar Complement Consistency Loss: (P(act) + P(stay) - 1)^2
        # Under excluded middle, exactly one of act or stay must be endorsed.
        loss_comp = torch.mean((p_act + p_stay - 1.0) ** 2)

        # 3. Dynamic Consequence-Sensitive Utility Ranking Loss
        # Target margin reflects the underlying utility difference Delta U (e.g. N_saved - N_sacrificed),
        # avoiding rigid overconfidence on close decisions while gracefully handling ties (Delta U == 0).
        loss_mono = torch.tensor(0.0, device=p_act.device)
        if stakes_diff is not None:
            stakes_diff = stakes_diff.view(-1)
            # Differentiate strictly positive, strictly negative, and tied utility outcomes
            pos_mask = stakes_diff > 0
            if pos_mask.any():
                score_diff = p_act[pos_mask] - p_stay[pos_mask]
                # Dynamic target margin: saturating smoothly with utility magnitude
                dynamic_margin = torch.clamp(stakes_diff[pos_mask] / 10.0, max=1.0) * self.margin
                loss_mono = torch.mean(F.relu(dynamic_margin - score_diff))

        total_loss = loss_sup + self.lambda_comp * loss_comp + self.lambda_mono * loss_mono

        metrics = {
            "loss_total": total_loss.item(),
            "loss_sup": loss_sup.item(),
            "loss_comp": loss_comp.item(),
            "loss_mono": loss_mono.item(),
            "mean_polarity_gap": torch.mean(torch.abs(p_act + p_stay - 1.0)).item(),
            "contradiction_rate": (
                ((p_act > 0.5) & (p_stay > 0.5)) | ((p_act < 0.5) & (p_stay < 0.5))
            ).float().mean().item(),
        }

        return total_loss, metrics


class UnanchoredContrastiveLoss(nn.Module):
    """Demonstration of naive contrastive penalty prone to trivial collapse.

    L_contrast = (s_act + s_stay)^2
    Minimizing this alone without anchoring drives logits s -> 0 and P -> 0.5.
    Provided for educational and diagnostic ablation tests.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logit_act: torch.Tensor, logit_stay: torch.Tensor) -> torch.Tensor:
        return torch.mean((logit_act + logit_stay) ** 2)
