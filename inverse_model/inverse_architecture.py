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
        # 正确形式：codes[B,p,w] 与 basis[B,L,w] 先对 w 做内积 → [B,p,L]，再对 p 求和
        # einsum("bpw,blw->bpl") 等价于矩阵乘法 codes @ basis.transpose(-1,-2)
        # 此写法避免 p 维度被直接求和消除（之前 bpw,blw->bl 等价于 p=1）
        out = torch.einsum("bpw, blw -> bpl", codes, basis)  # [B, p, L]
        out = out.mean(dim=1)                                 # [B, L]，对 p 个模式平均
        return out.unsqueeze(-1)           # [B, L, 1]
        out = out.mean(dim=1)                                 # [B, L]，对 p 个模式平均
        return out.unsqueeze(-1)           # [B, L, 1]