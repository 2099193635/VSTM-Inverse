"""BranchEncoder — 将时域加速度序列编码为 DeepONet 系数向量组.

输入：a_c(t)  [B, T, n_sensors]
输出：codes   [B, branch_modes, width]

结构：
  Conv1d lift → FNOBlock1d × branch_layers → AdaptiveAvgPool1d → 转置
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from inverse_config import InverseConfig


class SpectralConv1d(nn.Module):
    """1D 谱卷积层（复用 pino_architecture.py 的等价实现）.

    若仓库中已有 pino_model.pino_architecture.SpectralConv1d，
    直接替换此类的 import 即可：
        from pino_model.pino_architecture import SpectralConv1d
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                     # [B, C, T//2+1]
        out_ft = torch.zeros(B, self.weights.shape[1], T // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bci,coi->boi", x_ft[:, :, :self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=T, dim=-1)           # [B, C_out, T]


class FNOBlock1d(nn.Module):
    """FNO 基本块：谱卷积 + 残差 skip + 激活."""

    def __init__(self, width: int, modes: int, activation: str = "gelu") -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.skip = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.LayerNorm(width)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, width, T]
        out = self.spectral(x) + self.skip(x)      # 谱路径 + 跳跃连接
        out = out.transpose(1, 2)                   # [B, T, width]
        out = self.norm(out)
        out = out.transpose(1, 2)                   # [B, width, T]
        return self.act(out)


class BranchEncoder(nn.Module):
    """将时域加速度序列编码为 DeepONet 系数向量组.

    Args:
        cfg: InverseConfig

    Inputs:
        a_c: [B, T, n_sensors]  传感器加速度序列（时域）

    Outputs:
        codes: [B, branch_modes, width]  p 个宽度为 w 的系数向量
    """

    def __init__(self, cfg: InverseConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. 输入通道提升：n_sensors → width
        self.lift = nn.Conv1d(cfg.n_sensors, cfg.width, kernel_size=1)

        # 2. FNO 特征提取：捕捉时域频谱全局特征
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(cfg.width, cfg.fno_modes, cfg.activation)
            for _ in range(cfg.branch_layers)
        ])

        # 3. 全局聚合：T → branch_modes（固定长度，不依赖输入 T）
        self.pool = nn.AdaptiveAvgPool1d(cfg.branch_modes)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, a_c: torch.Tensor) -> torch.Tensor:
        # a_c: [B, T, n_sensors]
        x = a_c.transpose(1, 2)          # [B, n_sensors, T]
        x = self.lift(x)                  # [B, width, T]

        for block in self.fno_blocks:
            x = block(x)                  # [B, width, T]

        x = self.dropout(x)
        x = self.pool(x)                  # [B, width, branch_modes]
        codes = x.transpose(1, 2)        # [B, branch_modes, width]
        return codes
