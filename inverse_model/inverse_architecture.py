'''
Author: 2099193635 2099193635@qq.com
Date: 2026-04-07 16:17:46
LastEditors: 2099193635 2099193635@qq.com
LastEditTime: 2026-04-22 07:34:28
FilePath: /VTCM_PYTHON/inverse_model/inverse_architecture.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from inverse_config import InverseConfig
from branch_encoder import BranchEncoder
from trunk_decoder import TrunkDecoder
from physics_conditioner import FiLMConditioner

class InverseOperator(nn.Module):
    def __init__(self, cfg: InverseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.branch = BranchEncoder(cfg)
        self.trunk = TrunkDecoder(cfg)
        self.film = FiLMConditioner(cfg)

    def forward(
        self,
        a_c: torch.Tensor,
        x_query: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        # ── Branch： ──────────────────────────────────────────────
        codes = self.branch(a_c)           # [B, p, w]

        # ── Trunk：─────────────────────────────────────────────
        basis = self.trunk(x_query)        # [B, L, w]

        # ── FiLM： ────────────────────────────────────────────────
        basis = self.film(cond, basis)     # [B, L, w]

        # ── DeepONet  einsum ────────────────────────
        # codes [B, p, w] × basis [B, L, w] → [B, L]
        out = torch.einsum("bpw, blw -> bl", codes, basis)  # [B, L]
        out = out / self.cfg.branch_modes                    # 归一化（/ p）
        return out.unsqueeze(-1)           # [B, L, 1]