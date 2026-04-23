'''
Author: 2099193635 2099193635@qq.com
Date: 2026-04-07 16:15:41
LastEditors: 2099193635 2099193635@qq.com
LastEditTime: 2026-04-22 13:05:20
FilePath: /VTCM_PYTHON/inverse_model/trunk_decoder.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
"""TrunkDecoder — 将空间查询坐标映射为 DeepONet 基函数.

输入：x_query  [B, L]  归一化空间位置 ∈ [0, 1]
输出：basis    [B, L, width]

结构：
  Fourier PE → MLP (trunk_layers 层) → LayerNorm
"""
import math

import torch
import torch.nn as nn

from inverse_config import InverseConfig


class FourierPE(nn.Module):
    """确定性 Fourier 位置编码.

    PE(x) = [sin(2π·1·x), cos(2π·1·x),
             sin(2π·2·x), cos(2π·2·x),
             ...
             sin(2π·K·x), cos(2π·K·x)]  → [B, L, 2K]

    K=8 时，覆盖的波长范围（假设轨道段长 240m）：
      k=1 → λ=240m,  k=2 → 120m, ..., k=8 → 30m
    高铁轨道不平顺主要波长 1.5–120m，fpe_freqs=8 基本覆盖。
    """

    def __init__(self, n_freqs: int) -> None:
        super().__init__()
        # 注册为 buffer（不参与梯度，但随模型保存/移动设备）
        freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32)  # [K]
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]，值域 [0, 1]
        x = x.unsqueeze(-1)                         # [B, L, 1]
        angles = 2.0 * math.pi * x * self.freqs     # [B, L, K]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # → [B, L, 2K]


class TrunkDecoder(nn.Module):
    """将空间查询坐标映射为基函数矩阵.

    Args:
        cfg: InverseConfig

    Inputs:
        x_query: [B, L]  归一化空间位置

    Outputs:
        basis: [B, L, width]  每个查询点处的 p 个基函数值
    """

    def __init__(self, cfg: InverseConfig) -> None:
        super().__init__()
        self.pe = FourierPE(cfg.fpe_freqs)
        pe_dim = 2 * cfg.fpe_freqs  # 2K

        act = nn.GELU() if cfg.activation == "gelu" else nn.ReLU()

        # MLP：pe_dim → trunk_hidden × trunk_layers → width
        layers: list[nn.Module] = [nn.Linear(pe_dim, cfg.trunk_hidden), act]
        for _ in range(cfg.trunk_layers - 1):
            layers += [nn.Linear(cfg.trunk_hidden, cfg.trunk_hidden), act]
        layers.append(nn.Linear(cfg.trunk_hidden, cfg.width))

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, x_query: torch.Tensor) -> torch.Tensor:
        # x_query: [B, L]
        pe = self.pe(x_query)          # [B, L, 2K]
        basis = self.mlp(pe)           # [B, L, width]
        return self.norm(basis)        # [B, L, width]
