'''
Author: 2099193635 2099193635@qq.com
Date: 2026-04-07 16:17:40
LastEditors: 2099193635 2099193635@qq.com
LastEditTime: 2026-04-22 07:34:00
FilePath: /VTCM_PYTHON/inverse_model/physics_conditioner.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from typing import Optional, Any
from inverse_config import InverseConfig

def compute_modal_properties(
        M: torch.Tensor,
        K: torch.Tensor,
        C: torch.Tensor,
        n_modal: int = 8,
        vehicle_params: Optional[Any] = None,
        rayleigh_alpha: Optional[float] = None,
        rayleigh_beta: Optional[float] = None,
)-> tuple[torch.Tensor, torch.Tensor]:
    try:
        L = torch.linalg.cholesky(M)
        K_tilde = torch.linalg.solve_triangular(L, K, upper=False)
        K_tilde = torch.linalg.solve_triangular(L, K_tilde.T, upper=False).T
        eigvals = torch.linalg.eigvalsh(K_tilde)
        eigvals = eigvals[:n_modal].clamp(min=0.0)
        omega_n = eigvals.sqrt()
        natural_freqs = omega_n / (2 * torch.pi)
    except Exception:
        natural_freqs = torch.zeros(n_modal, device=M.device, dtype=M.dtype)
        omega_n = natural_freqs * (2 * torch.pi)

    try:
        alpha = rayleigh_alpha
        beta = rayleigh_beta

        if vehicle_params is not None:
            if alpha is None:
                alpha = getattr(vehicle_params, "rayleigh_alpha", None)
            if alpha is None:
                alpha = getattr(vehicle_params, "alpha", None)

            if beta is None:
                beta = getattr(vehicle_params, "rayleigh_beta", None)
            if beta is None:
                beta = getattr(vehicle_params, "beta", None)

        alpha = float(0.0 if alpha is None else alpha)
        beta = float(0.0 if beta is None else beta)

        omega_n_safe = omega_n.clamp(min=1e-6)
        damping_ratios = (alpha + beta * omega_n_safe ** 2) / (2.0 * omega_n_safe)
        damping_ratios = damping_ratios.clamp(0.0, 1.0)
    except Exception:
        damping_ratios = torch.zeros(n_modal, device=M.device, dtype=M.dtype)

    return natural_freqs, damping_ratios

class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioner, which applies an affine transformation to the basis features, where the scale and shift parameters are generated from the conditioning vector through an MLP.

    Args:
        cfg: InverseConfig
    
    Input:
        cond: [B, n_cond]       
        basis: [B, L, width]
    Output:
        basis': [B, L, width]
    """
    def __init__(self, cfg: InverseConfig):
        super().__init__()

        hidden = 128
        self.width = cfg.width

        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_cond, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, 2 * self.width)
        )
    
    def forward(self,
                cond: torch.Tensor,
                basis: torch.Tensor
        )-> torch.Tensor:
        # cond: [B, n_cond]
        # basis: [B, L, width]
        out = self.mlp(cond)                                 # [B, 2*width]
        raw_scale, shift = out.chunk(2, dim=-1)              # [B, width], [B, width]
        scale = torch.tanh(raw_scale).unsqueeze(1)           # [B, 1, width]
        shift = shift.unsqueeze(1)                           # [B, 1, width]

        return basis * (1 + scale) + shift                   # [B, L, width]