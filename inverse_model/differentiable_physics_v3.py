"""
differentiable_physics_v3.py
重新设计的物理信息层，解决两个核心问题：
  1. 细节波动学不到  → SpectralWeightedLoss：频率加权 MSE，对高频分量额外惩罚
  2. 更稳定的物理约束 → FRFForwardConsistencyLoss：正向计算车体响应与实测对比
     原理：若 z_pred 正确，则 z_pred → FRF → a_pred 应与传感器 a_true 的归一化
     频谱形状吻合。使用归一化谱形而非绝对幅值，避免模态残余幅值未收敛时损失爆炸。

为何 v2 的 DynamicEquationLoss (dyn) 不起作用：
  1. 量纲问题：残差 R = ÿ + 2ζωẏ + ω²y - ω²z，其中 ω~63 rad/s，量级放大 ~ω²~4000，
     归一化不足导致损失天然停留在 ~1，与数据损失梯度量级相差数量级。
  2. 梯度信号弱：lambda_dyn 取小值以防爆炸，但于是物理项对权重更新贡献可忽略。
  3. 积分漂移：从加速度积分位移本身存在低频累积误差（即使 FFT 积分也受窗口效应影响），
     使残差项含有大量非物理低频噪声。

v3 设计原则：
  - FRF 约束：在频域操作，天然避免积分误差，量级由归一化控制，始终在 [0,2] 范围内
  - 频谱加权损失：在数据损失中直接解决高频学习不足，不增加额外物理层复杂度
  - 模式：none / frf / spectral / both
      none     → SpectralWeightedLoss（仅频谱加权数据损失）
      frf      → SpectralWeightedLoss + FRF 正向一致性
      spectral → SpectralWeightedLoss（同 none，但明确标记）
      both     → 同 frf（别名）

支持模式简化为两个正交开关：
  use_spectral_data_loss: 是否用频谱加权替换纯 MSE（默认 True，改善高频）
  use_frf_physics:        是否加 FRF 正向一致性物理约束（默认 False）
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1. 频率加权数据损失（解决细节波动学不到的问题）
# ─────────────────────────────────────────────────────────────────────────────

class SpectralWeightedLoss(nn.Module):
    """
    频率加权 MSE 损失，在时域 MSE 基础上叠加频谱域加权误差项。

    总损失：
        L = L_td + beta * L_spec
        L_td   = MSE(pred, target)                            （时域均方误差）
        L_spec = mean(w(f) * |FFT(pred) - FFT(target)|²)     （加权频谱误差）
        w(f)   = 1 + gamma * (f / f_max)^alpha                （高频权重递增）

    为什么有效：
        FNO 的谱卷积截断在低模态数处（如 num_fno_modes=12），高频分量仅靠残差路径
        传播，梯度非常微弱。时域 MSE 中高频振荡的绝对误差相对幅值较小，被低频主导。
        频谱加权损失直接对高频误差施加更大梯度，迫使模型关注细节。

    Args:
        beta:   频谱项权重（推荐 0.1~1.0，过大会使训练不稳定）
        gamma:  高频强调因子（推荐 2.0~10.0）
        alpha:  频率权重的幂次（1.0=线性，2.0=二次）
    """

    def __init__(self, beta: float = 0.5, gamma: float = 5.0, alpha: float = 1.0) -> None:
        super().__init__()
        self.beta  = beta
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred, target: [B, n_channels, T]  (channel-first)
        """
        # 时域 MSE
        loss_td = F.mse_loss(pred, target)

        # 频谱加权 MSE
        B, C, T = pred.shape
        P = torch.fft.rfft(pred,   dim=-1)   # [B, C, F]
        Q = torch.fft.rfft(target, dim=-1)   # [B, C, F]
        F_len = P.shape[-1]

        # 频率权重：f_idx/F_len ∈ [0,1]，高频权重更大
        f_idx = torch.arange(F_len, device=pred.device, dtype=pred.dtype) / F_len
        w = 1.0 + self.gamma * f_idx.pow(self.alpha)              # [F]

        # 归一化（使频谱项量级接近时域 MSE）
        norm = T * C * B
        loss_spec = (w * (P - Q).abs().pow(2)).sum() / norm

        return loss_td + self.beta * loss_spec


# ─────────────────────────────────────────────────────────────────────────────
# 2. FRF 正向一致性物理约束（正向计算车体响应 → 与实测对比）
# ─────────────────────────────────────────────────────────────────────────────

class _ModalFRF(nn.Module):
    """
    内部模态 FRF 层，包含可学习模态参与因子 R。

    H(ω) = Σᵢ  -ω² · Rᵢ / (ωᵢ² - ω² + 2j·ζᵢ·ωᵢ·ω)

    Rᵢ [n_sensors, n_modal] 可学习，用 tanh 约束幅值，防止梯度爆炸。
    初始化：Rᵢ = R_max · tanh(1/(1+i)^0.5)，越高阶模态参与越小。
    """

    def __init__(self, n_modal: int, n_sensors: int, R_max: float = 5.0) -> None:
        super().__init__()
        self.n_modal   = n_modal
        self.n_sensors = n_sensors
        self.R_max     = R_max

        init_R = torch.zeros(n_sensors, n_modal)
        for i in range(n_modal):
            init_R[:, i] = 1.0 / (1.0 + i) ** 0.5
        self.raw_R = nn.Parameter(torch.atanh((init_R / R_max).clamp(-0.999, 0.999)))

    @property
    def R(self) -> Tensor:
        return torch.tanh(self.raw_R) * self.R_max   # [n_sensors, n_modal]

    def forward(
        self,
        Z_kappa:     Tensor,   # [B, F]     复数空间频谱
        omega_rad:   Tensor,   # [B, F]     对应角频率 [rad/s]
        nat_freqs:   Tensor,   # [B, n_modal]
        damp_ratios: Tensor,   # [B, n_modal]
    ) -> Tensor:               # [B, F, n_sensors] 复数加速度谱
        B, F = Z_kappa.shape

        omega_q = omega_rad.unsqueeze(-1).to(torch.complex64)     # [B, F, 1]
        omega_i = nat_freqs.unsqueeze(1).to(torch.complex64)      # [B, 1, n_modal]
        zeta_i  = damp_ratios.unsqueeze(1).to(torch.complex64)    # [B, 1, n_modal]
        R_i     = self.R.unsqueeze(0)                             # [1, n_s, n_modal]

        # 分母 [B, F, n_modal]
        denom = (omega_i ** 2 - omega_q ** 2
                 + 2j * zeta_i * omega_i * omega_q)
        denom = torch.where(denom.abs() < 1e-8,
                            torch.full_like(denom, 1e-8), denom)

        # H [B, F, n_sensors]
        denom_4d = denom.unsqueeze(2)                                # [B, F, 1, n_modal]
        R_4d     = R_i.unsqueeze(1).to(torch.complex64)             # [1, 1, n_s, n_modal]
        H = -(omega_q ** 2) * (R_4d / denom_4d).sum(dim=-1)        # [B, F, n_s]

        # A = H · Z  [B, F, n_sensors]
        return H * Z_kappa.unsqueeze(-1).to(torch.complex64)


class FRFForwardConsistencyLoss(nn.Module):
    """
    FRF 正向一致性损失。

    物理流程：
        z_pred [B,L]  →  FFT  →  Z_κ(ω)
        κ → ω = 2πκv_x  （空间频率变换为时间频率）
        Z_κ × H(ω; ω_n, ζ, R)  →  A_pred(ω) [B, F, n_s]
        IFFT  →  a_pred [B, T, n_s]

    损失计算（归一化频谱形状对比）：
        - 计算 a_pred 和 a_true 的单边幅值谱
        - 各自按样本/传感器的 L2 范数归一化（消除绝对幅值差异）
        - 对比归一化后的谱形 → 约束频率结构，而非绝对值
        这样即使初始 R 未收敛，损失仍能提供稳定的方向性梯度。

    为什么比 DynamicEquationLoss 更稳定：
        1. 全程频域操作，无积分误差
        2. 归一化消除幅值不匹配
        3. 损失天然 ∈ [0, 2]，无需 lambda 调参
        4. 梯度直接反传到 z_pred，物理约束有效

    Args:
        n_modal:    模态数
        n_sensors:  传感器数
        dx:         空间采样步长 [m]
        dt:         时间采样步长 [s]（用于 IFFT 输出长度匹配）
        R_max:      模态参与因子上界
    """

    def __init__(
        self,
        n_modal:   int   = 8,
        n_sensors: int   = 3,
        dx:        float = 0.25,
        dt:        float = 0.005,
        R_max:     float = 5.0,
    ) -> None:
        super().__init__()
        self.dx  = dx
        self.dt  = dt
        self.frf = _ModalFRF(n_modal, n_sensors, R_max)

    def forward(
        self,
        z_pred:      Tensor,   # [B, L]
        a_c_true:    Tensor,   # [B, T, n_sensors]
        nat_freqs:   Tensor,   # [B, n_modal]
        damp_ratios: Tensor,   # [B, n_modal]
        vx_mps:      Tensor,   # [B]
    ) -> Tensor:               # 标量损失
        if z_pred.dim() == 3:
            z_pred = z_pred.squeeze(-1)
        B, L = z_pred.shape
        T = a_c_true.shape[1]

        # Step 1：空间 FFT
        Z_kappa = torch.fft.rfft(z_pred, dim=-1)                          # [B, F]
        kappa   = torch.fft.rfftfreq(L, d=self.dx, device=z_pred.device)  # [F]

        # Step 2：κ → ω
        omega_rad = 2.0 * math.pi * kappa.unsqueeze(0) * vx_mps.unsqueeze(1)  # [B, F]

        # Step 3：模态 FRF
        A_omega = self.frf(Z_kappa, omega_rad, nat_freqs, damp_ratios)    # [B, F, n_s]

        # Step 4：IFFT → 时域 [B, T, n_s]
        a_pred = torch.fft.irfft(
            A_omega.permute(0, 2, 1), n=T, dim=-1
        ).transpose(1, 2).contiguous()                                     # [B, T, n_s]

        # Step 5：归一化谱形对比
        # 计算时域单边幅值谱（沿 T 轴）
        A_pred = torch.fft.rfft(a_pred,   dim=1).abs()                    # [B, Ft, n_s]
        A_true = torch.fft.rfft(a_c_true, dim=1).abs()                    # [B, Ft, n_s]

        # 按样本+传感器归一化（L2 norm，消除绝对幅值差异）
        A_pred_n = A_pred / (A_pred.norm(dim=1, keepdim=True).clamp(min=1e-8))
        A_true_n = A_true / (A_true.norm(dim=1, keepdim=True).clamp(min=1e-8))

        return (A_pred_n - A_true_n).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3. 统一调度接口 PhysicsLayerV3
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsLayerV3(nn.Module):
    """
    v3 统一物理信息层。

    两个正交开关：
        use_spectral:  True → 用 SpectralWeightedLoss 替换数据项纯 MSE（默认 True）
        use_frf:       True → 额外添加 FRF 正向一致性物理损失（默认视 mode 决定）

    物理模式（--physics_mode）：
        none     → use_spectral=True,  use_frf=False   只改善高频学习
        spectral → use_spectral=True,  use_frf=False   同 none
        frf      → use_spectral=True,  use_frf=True    频谱数据损失 + 物理约束
        both     → use_spectral=True,  use_frf=True    同 frf（别名）

    Args:
        mode:             物理模式
        n_modal:          FRF 模态数
        n_sensors:        传感器数
        dx, dt:           空间/时间步长
        spectral_beta:    频谱数据损失中频谱项权重
        spectral_gamma:   高频强调因子
        spectral_alpha:   频率权重幂次
        frf_R_max:        模态参与因子幅值上界
    """

    def __init__(
        self,
        mode:           str   = "frf",
        n_modal:        int   = 8,
        n_sensors:      int   = 3,
        dx:             float = 0.25,
        dt:             float = 0.005,
        spectral_beta:  float = 0.5,
        spectral_gamma: float = 5.0,
        spectral_alpha: float = 1.0,
        frf_R_max:      float = 5.0,
    ) -> None:
        super().__init__()
        self.mode        = mode
        self.use_frf     = mode in ("frf", "both")
        self.use_spectral = True  # 始终开启频谱加权数据损失

        self.spectral_loss_fn = SpectralWeightedLoss(
            beta=spectral_beta, gamma=spectral_gamma, alpha=spectral_alpha
        )
        if self.use_frf:
            self.frf_loss_fn = FRFForwardConsistencyLoss(
                n_modal=n_modal, n_sensors=n_sensors,
                dx=dx, dt=dt, R_max=frf_R_max,
            )

    # ── 数据损失接口（替代 F.mse_loss）──────────────────────────────────────
    def compute_data_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        替代训练脚本中 F.mse_loss(pred, target) 的调用。
        pred, target: [B, n_channels, T]（channel-first）
        """
        return self.spectral_loss_fn(pred, target)

    # ── 物理约束损失接口 ─────────────────────────────────────────────────────
    def forward(
        self,
        z_pred:      Tensor,
        a_c_true:    Optional[Tensor] = None,
        nat_freqs:   Optional[Tensor] = None,
        damp_ratios: Optional[Tensor] = None,
        vx_mps:      Optional[Tensor] = None,
        lambda_frf:  float = 1.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Returns:
            loss:    FRF 物理约束损失（标量），若 use_frf=False 则为 0
            metrics: {"l_frf": float}
        """
        l_frf = torch.tensor(0.0, device=z_pred.device)

        if self.use_frf:
            assert all(x is not None for x in [a_c_true, nat_freqs, damp_ratios, vx_mps]), \
                "frf 模式需要 a_c_true, nat_freqs, damp_ratios, vx_mps"
            l_frf = self.frf_loss_fn(
                z_pred, a_c_true, nat_freqs, damp_ratios, vx_mps
            )

        return lambda_frf * l_frf, {"l_frf": l_frf.item()}
