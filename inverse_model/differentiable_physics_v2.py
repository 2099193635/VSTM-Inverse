"""
differentiable_physics_v2.py
新版可微物理信息层，包含两类约束：

  1. DynamicEquationLoss
     车辆-轨道耦合动力学方程 ODE 残差（时域模态叠加）。
     对每阶模态 i：
         ÿ_i(t) + 2ζ_i ω_i ẏ_i(t) + ω_i² y_i(t) = ω_i² z_irr(t)
     其中 z_irr(t) 由预测的空间不平顺经速度变换得到。

  2. IrregularityPSDLoss
     轨道不平顺空间功率谱密度（PSD）约束。
     理论模型（中国标准 TB/T 3352 / Sato 公式）：
         S(Ω) = A_v · Ω_c² / [(Ω² + Ω_r²)(Ω² + Ω_c²)]
     在 dB（对数）尺度下约束预测不平顺的谱形与幅值。

  3. PhysicsLayerV2
     统一调度接口，模式：none / dyn / psd / both
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1. 动力学方程残差
# ─────────────────────────────────────────────────────────────────────────────

class DynamicEquationLoss(nn.Module):
    """
    车辆-轨道耦合动力学 ODE 残差损失（时域，模态叠加法）。

    物理方程（每阶模态的基础激励形式）：
        ÿ(t) + 2ζᵢωᵢẏ(t) + ωᵢ²y(t) = ωᵢ² · z_irr(t)

    其中：
        ωᵢ      : 第 i 阶固有圆频率 [rad/s]（来自条件向量）
        ζᵢ      : 第 i 阶阻尼比（来自条件向量）
        z_irr(t): 时域不平顺激励（由空间不平顺经 x = v·t 变换得到）
        ÿ(t)    : 传感器实测加速度（已知）
        ẏ(t)    : 速度（由加速度谱积分得到）
        y(t)    : 位移（由加速度谱积分得到）

    残差：
        R(t) = ÿ(t) + 2ζωẏ(t) + ω²y(t) − ω² z_irr(t)

    损失 = 各模态残差均方 / n_modal

    Args:
        dt: 时间采样步长 [s]
        dx: 空间采样步长 [m]
    """

    def __init__(self, dt: float = 0.005, dx: float = 0.25) -> None:
        super().__init__()
        self.dt = dt
        self.dx = dx

    # ── 谱积分：加速度 → 速度 / 位移 ─────────────────────────────────────────
    def _spectral_integrate(self, a: Tensor) -> tuple[Tensor, Tensor]:
        """
        基于 FFT 的谱积分，避免时域积分的累积误差。
        a:       [B, T, n_s]
        returns: (vel [B, T, n_s], disp [B, T, n_s])
        """
        B, T, n_s = a.shape
        # 变换到 [B*n_s, T] 进行批量 FFT
        a_flat = a.permute(0, 2, 1).reshape(B * n_s, T)      # [B*n_s, T]

        A     = torch.fft.rfft(a_flat, dim=-1)                # [B*n_s, F]
        freq  = torch.fft.rfftfreq(T, d=self.dt, device=a.device)   # [F]
        omega = 2.0 * math.pi * freq                          # [F] rad/s

        # 防止 DC 分量除零（积分后置零）
        omega_nz = omega.clone()
        omega_nz[0] = 1.0

        # 积分一次 → 速度：V(ω) = A(ω) / (jω)
        V = A / (1j * omega_nz.to(torch.complex64))
        # 积分两次 → 位移：D(ω) = A(ω) / (−ω²)
        D = A / (-(omega_nz ** 2).to(torch.complex64))

        # 清零 DC（消除积分漂移）
        V[..., 0] = 0.0
        D[..., 0] = 0.0

        vel  = torch.fft.irfft(V, n=T, dim=-1)               # [B*n_s, T]
        disp = torch.fft.irfft(D, n=T, dim=-1)               # [B*n_s, T]

        vel  = vel .reshape(B, n_s, T).permute(0, 2, 1)      # [B, T, n_s]
        disp = disp.reshape(B, n_s, T).permute(0, 2, 1)      # [B, T, n_s]
        return vel, disp

    # ── 空间 → 时域不平顺变换 ────────────────────────────────────────────────
    def _spatial_to_temporal(
        self, z_pred: Tensor, vx_mps: Tensor, T_target: int
    ) -> Tensor:
        """
        z_pred [B, L] (空间域，步长 dx) → z_irr [B, T_target] (时域)
        坐标变换：x = v · t，采用线性插值。
        """
        B, L = z_pred.shape
        # 时间轴对应的空间坐标
        t = torch.arange(
            T_target, device=z_pred.device, dtype=z_pred.dtype
        ) * self.dt                                           # [T]
        x   = vx_mps.unsqueeze(1) * t.unsqueeze(0)           # [B, T]

        # 连续空间索引，约束在 [0, L-1] 范围
        idx   = (x / self.dx).clamp(0.0, float(L - 1) - 1e-6)  # [B, T]
        idx_f = idx.long()                                    # [B, T]
        idx_c = (idx_f + 1).clamp(max=L - 1)                 # [B, T]
        frac  = idx - idx_f.float()                           # [B, T]

        z_f = z_pred.gather(1, idx_f)                        # [B, T]
        z_c = z_pred.gather(1, idx_c)                        # [B, T]
        return z_f * (1.0 - frac) + z_c * frac              # [B, T]

    # ── 前向计算 ──────────────────────────────────────────────────────────────
    def forward(
        self,
        z_pred:      Tensor,   # [B, L]  预测的空间不平顺
        a_c_true:    Tensor,   # [B, T, n_s]  实测加速度
        nat_freqs:   Tensor,   # [B, n_modal]  固有圆频率 [rad/s]
        damp_ratios: Tensor,   # [B, n_modal]  阻尼比
        vx_mps:      Tensor,   # [B]  行驶速度 [m/s]
    ) -> Tensor:
        B, T, n_s = a_c_true.shape
        n_modal   = nat_freqs.shape[1]

        # Step 1：空间不平顺 → 时域激励
        z_t    = self._spatial_to_temporal(z_pred, vx_mps, T)  # [B, T]
        z_t_3d = z_t.unsqueeze(-1)                             # [B, T, 1]

        # Step 2：谱积分得速度、位移
        vel, disp = self._spectral_integrate(a_c_true)         # [B, T, n_s]

        # Step 3：逐模态 ODE 残差
        total = torch.zeros(1, device=z_pred.device, dtype=z_pred.dtype).squeeze()
        for i in range(n_modal):
            omega_i = nat_freqs[:, i].view(B, 1, 1)           # [B, 1, 1]
            zeta_i  = damp_ratios[:, i].view(B, 1, 1)         # [B, 1, 1]

            # R(t) = ÿ + 2ζω ẏ + ω² y − ω² z_irr
            R = (
                a_c_true
                + 2.0 * zeta_i * omega_i * vel
                + omega_i ** 2 * disp
                - omega_i ** 2 * z_t_3d
            )                                                  # [B, T, n_s]
            total = total + (R ** 2).mean()

        return total / n_modal


# ─────────────────────────────────────────────────────────────────────────────
# 2. 不平顺谱密度约束
# ─────────────────────────────────────────────────────────────────────────────

class IrregularityPSDLoss(nn.Module):
    """
    轨道不平顺空间功率谱密度（PSD）约束损失。

    理论谱密度模型（中国标准 TB/T 3352 / Sato 公式，竖向不平顺）：
        S(Ω) = A_v · Ω_c² / [(Ω² + Ω_r²)(Ω² + Ω_c²)]   [m²·rad/m]

    参数说明：
        Ω     : 空间频率 [rad/m]
        A_v   : 谱幅值参数，与轨道等级相关
                  优：1.5e-7  良：4.4e-7  一般：1.3e-6
        Ω_c   : 截止空间频率（0.8246 rad/m，对应波长约 7.6 m）
        Ω_r   : 参考空间频率（0.0206 rad/m，对应波长约 305 m）

    损失在对数坐标（dB）下计算，使低频高频段均衡加权。

    支持可学习的幅值比例因子（learnable_scale=True）：
        网络可以调整幅值而约束谱形，避免因轨道等级不匹配而导致损失偏大。

    Args:
        dx:              空间采样步长 [m]
        A_v:             理论谱幅值参数 [m²·rad/m]
        Omega_c:         截止空间频率 [rad/m]
        Omega_r:         参考空间频率 [rad/m]
        freq_range:      有效空间频率范围 (Ω_min, Ω_max) [rad/m]
        learnable_scale: 是否使用可学习的对数幅值比例因子
    """

    def __init__(
        self,
        dx:              float = 0.25,
        A_v:             float = 4.4e-7,
        Omega_c:         float = 0.8246,
        Omega_r:         float = 0.0206,
        freq_range:      tuple = (0.1, 10.0),
        learnable_scale: bool  = True,
    ) -> None:
        super().__init__()
        self.dx      = dx
        self.A_v     = A_v
        self.Omega_c = Omega_c
        self.Omega_r = Omega_r
        self.freq_range = freq_range

        # 可学习对数比例因子 log(scale)，初始化为 0（scale=1，不改变幅值）
        if learnable_scale:
            self.log_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_scale", torch.zeros(1))

    def _theoretical_psd(self, Omega: Tensor) -> Tensor:
        """
        计算标准谱密度值。
        Omega: [F] rad/m  →  S: [F]
        S(Ω) = A_v · Ω_c² / [(Ω² + Ω_r²)(Ω² + Ω_c²)]
        """
        Oc2 = self.Omega_c ** 2
        Or2 = self.Omega_r ** 2
        O2  = Omega ** 2
        S = self.A_v * Oc2 / ((O2 + Or2) * (O2 + Oc2) + 1e-30)
        # 可学习比例因子（log 空间保证正值）
        return S * self.log_scale.exp()

    def forward(self, z_pred: Tensor) -> Tensor:
        """
        z_pred: [B, L]  预测的空间不平顺
        """
        B, L = z_pred.shape

        # 周期图法估计 PSD
        Z = torch.fft.rfft(z_pred, dim=-1)                   # [B, F]
        Omega = (
            2.0 * math.pi
            * torch.fft.rfftfreq(L, d=self.dx, device=z_pred.device)
        )                                                     # [F] rad/m

        # 单边 PSD：S_est(Ω) ≈ |Z(Ω)|² · dx / (π · L)
        S_est = Z.abs() ** 2 * self.dx / (math.pi * L)       # [B, F]

        # 理论 PSD
        S_theory = self._theoretical_psd(Omega)               # [F]

        # 有效频率范围掩码
        Omega_min, Omega_max = self.freq_range
        mask = (Omega >= Omega_min) & (Omega <= Omega_max)    # [F]
        if mask.sum() < 2:
            return torch.tensor(0.0, device=z_pred.device)

        S_est_m    = S_est[:, mask].clamp(min=1e-30)         # [B, n_freq]
        S_theory_m = S_theory[mask].clamp(min=1e-30)         # [n_freq]

        # dB 尺度 MSE：log(S_est / S_theory)²
        log_ratio = torch.log(S_est_m) - torch.log(S_theory_m.unsqueeze(0))
        return (log_ratio ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3. 统一调度接口
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsLayerV2(nn.Module):
    """
    新版物理信息层调度接口。

    支持模式：
        none  ── 纯数据驱动
        dyn   ── 动力学 ODE 残差（需要 nat_freqs, damp_ratios, vx_mps, a_c_true）
        psd   ── 不平顺谱密度约束（仅需 z_pred）
        both  ── dyn + psd

    Args:
        mode:                物理模式 "none" / "dyn" / "psd" / "both"
        dt:                  时间步长 [s]（dyn/both 用）
        dx:                  空间步长 [m]
        A_v:                 不平顺谱幅值（psd/both 用）
        Omega_c:             截止空间频率 [rad/m]（psd/both 用）
        Omega_r:             参考空间频率 [rad/m]（psd/both 用）
        psd_freq_range:      有效空间频率范围 (Ω_min, Ω_max)（psd/both 用）
        psd_learnable_scale: PSD 幅值是否可学习（psd/both 用）
    """

    def __init__(
        self,
        mode:                str   = "dyn",
        dt:                  float = 0.005,
        dx:                  float = 0.25,
        A_v:                 float = 4.4e-7,
        Omega_c:             float = 0.8246,
        Omega_r:             float = 0.0206,
        psd_freq_range:      tuple = (0.1, 10.0),
        psd_learnable_scale: bool  = True,
    ) -> None:
        super().__init__()
        self.mode = mode

        if mode in ("dyn", "both"):
            self.dyn_layer = DynamicEquationLoss(dt=dt, dx=dx)

        if mode in ("psd", "both"):
            self.psd_layer = IrregularityPSDLoss(
                dx=dx,
                A_v=A_v,
                Omega_c=Omega_c,
                Omega_r=Omega_r,
                freq_range=psd_freq_range,
                learnable_scale=psd_learnable_scale,
            )

    def forward(
        self,
        z_pred:      Tensor,
        a_c_true:    Optional[Tensor] = None,   # [B, T, n_s]（dyn/both 必须）
        nat_freqs:   Optional[Tensor] = None,   # [B, n_modal]（dyn/both 必须）
        damp_ratios: Optional[Tensor] = None,   # [B, n_modal]（dyn/both 必须）
        vx_mps:      Optional[Tensor] = None,   # [B]（dyn/both 必须）
        lambda_dyn:  float = 1.0,
        lambda_psd:  float = 1.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Returns:
            loss:    标量物理损失
            metrics: {"l_dyn": float, "l_psd": float}
        """
        l_dyn = torch.tensor(0.0, device=z_pred.device)
        l_psd = torch.tensor(0.0, device=z_pred.device)

        if self.mode in ("dyn", "both"):
            assert all(x is not None for x in [a_c_true, nat_freqs, damp_ratios, vx_mps]), \
                "dyn 模式需要 a_c_true, nat_freqs, damp_ratios, vx_mps"
            l_dyn = self.dyn_layer(
                z_pred, a_c_true, nat_freqs, damp_ratios, vx_mps
            )

        if self.mode in ("psd", "both"):
            l_psd = self.psd_layer(z_pred)

        loss = lambda_dyn * l_dyn + lambda_psd * l_psd
        return loss, {"l_dyn": l_dyn.item(), "l_psd": l_psd.item()}
