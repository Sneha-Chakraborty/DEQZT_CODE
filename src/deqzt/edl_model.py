from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDLMLP(nn.Module):
    """
    Lightweight MLP for Evidential Deep Learning (EDL).

    It outputs non-negative "evidence" per class, which is converted to Dirichlet
    concentration parameters:
        alpha = evidence + 1
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: Sequence[int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hs = list(hidden_sizes)
        layers: List[nn.Module] = []
        d = int(input_dim)
        for h in hs:
            layers.append(nn.Linear(d, int(h)))
            layers.append(nn.ReLU())
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = int(h)
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(d, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        logits = self.head(z)
        # Softplus for strictly positive evidence (numerically stable)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        return alpha


@dataclass
class EDLForward:
    alpha: torch.Tensor  # (B, K)
    probs: torch.Tensor  # (B, K)
    uncertainty: torch.Tensor  # (B,)

    @staticmethod
    def from_alpha(alpha: torch.Tensor) -> "EDLForward":
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / (S + 1e-9)
        K = alpha.shape[1]
        u = (float(K) / (S.squeeze(1) + 1e-9)).clamp(0.0, 1e9)
        return EDLForward(alpha=alpha, probs=probs, uncertainty=u)
