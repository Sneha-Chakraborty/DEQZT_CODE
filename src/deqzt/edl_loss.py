from __future__ import annotations

from typing import Literal

import torch


def kl_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1) ) per-sample (shape: [B]).

    Dir(1) is the uniform Dirichlet prior.
    """
    # beta = ones
    beta = torch.ones((1, num_classes), device=alpha.device, dtype=alpha.dtype)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB_alpha + lnB_beta
    return kl.squeeze(1)


def edl_loss(
    alpha: torch.Tensor,
    target_onehot: torch.Tensor,
    num_classes: int,
    anneal: float,
    loss_type: Literal["log", "mse"] = "log",
) -> torch.Tensor:
    """
    EDL loss from Sensoy et al. (2018) commonly used in evidential classifiers.

    alpha: Dirichlet concentration (B, K), alpha = evidence + 1
    target_onehot: one-hot labels (B, K)
    anneal: KL annealing factor in [0,1]
    """
    S = torch.sum(alpha, dim=1, keepdim=True)  # (B,1)
    probs = alpha / (S + 1e-9)

    if loss_type == "mse":
        # MSE + variance term
        A = torch.sum((target_onehot - probs) ** 2, dim=1)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1.0) + 1e-9), dim=1)
        data_loss = A + B
    else:
        # Negative expected log likelihood:
        # E[log p_k] = digamma(alpha_k) - digamma(S)
        loglik = torch.sum(
            target_onehot * (torch.digamma(alpha) - torch.digamma(S)),
            dim=1,
        )
        data_loss = -loglik

    # Only penalize evidence for *incorrect* classes (alpha_tilde trick)
    alpha_tilde = (alpha - 1.0) * (1.0 - target_onehot) + 1.0
    kl = kl_dirichlet(alpha_tilde, num_classes=num_classes)

    return data_loss + float(anneal) * kl
