"""inverse_model/utils/vtcm_physics.py
VTCM 物理残差损失，供 PINO/FNO 等训练脚本共用。
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 路径兜底（以防直接导入）
_ROOT = Path(__file__).parent.parent.parent
_PHYSICSNEMO = str(_ROOT / "physicsnemo")
if _PHYSICSNEMO not in sys.path:
    sys.path.insert(0, _PHYSICSNEMO)
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physics_informed import VTCMFull

from .losses import denormalize_output


# ── 构建 VTCM 上下文 ──────────────────────────────────────────────────────────
def build_vtcm_context(vx_ms: float, device=None) -> dict:
    """
    构建 VTCMFull（dynamic_G=True）+ PhysicsInformer，返回训练所需常数字典。
    vx_ms: 列车速度（m/s）。
    """
    Mc, Mt, Mw   = 40000.0, 3200.0, 1450.0
    Ksz, Csz     = 240000.0, 60000.0
    Kpz, Cpz     = 1.18e6, 9800.0
    Ghertz       = 4.5e-8
    Lc, Lt       = 9.0, 1.2
    dt           = 1e-4
    F0           = Mc * 9.81 / 4 + Mt * 9.81 / 2 + Mw * 9.81

    vtcm_full = VTCMFull(
        Mc=Mc, Mt=Mt, Mw=Mw,
        Ksz=Ksz, Csz=Csz, Kpz=Kpz, Cpz=Cpz,
        G=Ghertz, F0=F0, dynamic_G=True,
    )
    pi_informer = PhysicsInformer(
        required_outputs=["car_body", "bogie_f", "bogie_r",
                          "nexus_w1", "nexus_w2", "nexus_w3", "nexus_w4"],
        equations=vtcm_full,
        grad_method="finite_difference",
        fd_dx=dt,
        device=str(device) if device is not None else "cpu",
    )
    return dict(
        vtcm_full=vtcm_full, pi_informer=pi_informer,
        Mc=Mc, Mt=Mt, Mw=Mw, Ksz=Ksz, Csz=Csz, Kpz=Kpz, Cpz=Cpz,
        Lc=Lc, Lt=Lt, vx_ms=float(vx_ms), dt=dt, Ghertz=Ghertz, F0=F0,
    )


# ── 时间差分 ──────────────────────────────────────────────────────────────────
def _fd1(x: torch.Tensor, dt: float) -> torch.Tensor:
    """中心差分一阶导，x: [B, C, T]。"""
    dx = torch.zeros_like(x)
    dx[..., 1:-1] = (x[..., 2:] - x[..., :-2]) / (2 * dt)
    dx[..., 0]    = (x[..., 1]  - x[..., 0])  / dt
    dx[..., -1]   = (x[..., -1] - x[..., -2]) / dt
    return dx


def _fd2(x: torch.Tensor, dt: float) -> torch.Tensor:
    """中心差分二阶导，x: [B, C, T]。"""
    ddx = torch.zeros_like(x)
    ddx[..., 1:-1] = (x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]) / (dt * dt)
    ddx[..., 0]    = ddx[..., 1]
    ddx[..., -1]   = ddx[..., -2]
    return ddx


def _fd2_1d(z: torch.Tensor, dt: float) -> torch.Tensor:
    """中心差分二阶导，z: [B, T]。"""
    interior = (z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]) / (dt * dt)
    return torch.cat([interior[:, :1], interior, interior[:, -1:]], dim=1)


# ── 时间平移（端点复制 + 亚步长插值）────────────────────────────────────────
def _shift_int(arr: torch.Tensor, steps: int) -> torch.Tensor:
    """arr: [B, T]，整数步平移，端点复制填充。"""
    if steps == 0:
        return arr
    steps = max(-arr.shape[1] + 1, min(arr.shape[1] - 1, steps))
    if steps > 0:
        return torch.cat([arr[:, :1].expand(-1, steps), arr[:, :-steps]], dim=1)
    s = -steps
    return torch.cat([arr[:, s:], arr[:, -1:].expand(-1, s)], dim=1)


def _shift_frac(arr: torch.Tensor, steps_f: float) -> torch.Tensor:
    """arr: [B, T]，浮点步亚步长线性插值平移。"""
    if steps_f == 0:
        return arr
    s_lo = int(steps_f) if steps_f >= 0 else -int(-steps_f)
    frac = steps_f - s_lo
    v_lo = _shift_int(arr, -s_lo) if s_lo != 0 else arr
    if abs(frac) < 1e-9:
        return v_lo
    v_hi = _shift_int(arr, -(s_lo + 1))
    return (1 - frac) * v_lo + frac * v_hi


def _nexus_delays(vtcm_ctx: dict) -> dict:
    """各轮对相对轮 4 的超前延迟（浮点步数）。"""
    V, dt = vtcm_ctx["vx_ms"], vtcm_ctx["dt"]
    Lc, Lt = vtcm_ctx["Lc"], vtcm_ctx["Lt"]
    return {"w1": 2 * (Lc + Lt) / V / dt,
            "w2": 2 * Lc / V / dt,
            "w3": 2 * Lt / V / dt}


def _dc_correct(Zw: torch.Tensor, Z0: torch.Tensor) -> torch.Tensor:
    """DC 修正：消除钢轨静弯挠度偏置，Z0 += mean_T(Zw - Z0)。"""
    return Z0 + (Zw - Z0).mean(dim=1, keepdim=True)


# ── VTCM 物理残差损失 ─────────────────────────────────────────────────────────
def vtcm_physics_loss(pred_norm: torch.Tensor, batch: dict,
                      device: torch.device, out_norm_stats: dict,
                      vtcm_ctx: dict, deriv_mix_alpha: float = 0.0,
                      nexus_weight: float = 0.1) -> tuple:
    """
    计算 VTCM 7 方程物理残差损失。

    Returns:
        loss_total     : scalar tensor，car_bogie + nexus_weight * nexus
        components     : dict of detached scalar tensors（各方程损失）
    """
    pi   = vtcm_ctx["pi_informer"]
    dt   = float(vtcm_ctx["dt"])
    d    = _nexus_delays(vtcm_ctx)

    # ── 反归一化 ──────────────────────────────────────────────────────────
    phys = denormalize_output(pred_norm, out_norm_stats, device)
    z0   = phys[:, 0, :]    # [B, T]
    x    = phys[:, 1:, :]   # [B, 7, T]

    # ── 导数混合（alpha=0: GT导数，alpha=1: 预测导数）─────────────────────
    alpha   = float(min(max(deriv_mix_alpha, 0.0), 1.0))
    gt_v    = batch["phys_v"].to(device, non_blocking=True).permute(0, 2, 1)
    gt_a    = batch["phys_a"].to(device, non_blocking=True).permute(0, 2, 1)
    bv      = (1.0 - alpha) * gt_v + alpha * _fd1(x, dt)
    ba      = (1.0 - alpha) * gt_a + alpha * _fd2(x, dt)

    # ── 各轮 Z0 时序对齐 ──────────────────────────────────────────────────
    def _get_precomputed(key):
        if key in batch:
            return batch[key].to(device, non_blocking=True).squeeze(-1)
        return None

    _z0w1 = _get_precomputed("phys_z0_w1"); Z0_w1 = _z0w1 if _z0w1 is not None else _shift_frac(z0, d["w1"])
    _z0w2 = _get_precomputed("phys_z0_w2"); Z0_w2 = _z0w2 if _z0w2 is not None else _shift_frac(z0, d["w2"])
    _z0w3 = _get_precomputed("phys_z0_w3"); Z0_w3 = _z0w3 if _z0w3 is not None else _shift_frac(z0, d["w3"])
    _z0w4 = _get_precomputed("phys_z0_w4"); Z0_w4 = _z0w4 if _z0w4 is not None else z0

    Zw1, Zw2, Zw3, Zw4 = x[:, 3, :], x[:, 4, :], x[:, 5, :], x[:, 6, :]

    # ── 构建 PI 输入 ──────────────────────────────────────────────────────
    names_pos = ("Zc", "Zt1", "Zt2", "Zw1", "Zw2", "Zw3", "Zw4")
    names_vel = ("Zc_v", "Zt1_v", "Zt2_v", "Zw1_v", "Zw2_v", "Zw3_v", "Zw4_v")
    names_acc = ("Zc_a", "Zt1_a", "Zt2_a", "Zw1_a", "Zw2_a", "Zw3_a", "Zw4_a")

    inp: dict = {}
    for i, n in enumerate(names_pos): inp[n] = x[:, i, :]
    for i, n in enumerate(names_vel): inp[n] = bv[:, i, :]
    for i, n in enumerate(names_acc): inp[n] = ba[:, i, :]

    inp["Z0_w1"] = _dc_correct(Zw1, Z0_w1);  inp["Z0_w1_a"] = _fd2_1d(Z0_w1, dt)
    inp["Z0_w2"] = _dc_correct(Zw2, Z0_w2);  inp["Z0_w2_a"] = _fd2_1d(Z0_w2, dt)
    inp["Z0_w3"] = _dc_correct(Zw3, Z0_w3);  inp["Z0_w3_a"] = _fd2_1d(Z0_w3, dt)
    inp["Z0_w4"] = _dc_correct(Zw4, Z0_w4);  inp["Z0_w4_a"] = _fd2_1d(Z0_w4, dt)

    # ── G_eff 反算 ────────────────────────────────────────────────────────
    VTCMFull.compute_Gwr_eff(
        inp,
        F0=vtcm_ctx["F0"], Kpz=vtcm_ctx["Kpz"], Cpz=vtcm_ctx["Cpz"],
        Mw=vtcm_ctx["Mw"], G_fallback=vtcm_ctx["Ghertz"],
        per_timestep=True,
    )

    # ── PI forward ────────────────────────────────────────────────────────
    res = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
           for k, v in pi.forward(inp).items()}

    # ── 有效区间 pad ──────────────────────────────────────────────────────
    T_seq  = x.shape[-1]
    pad_d  = {"car_body": 0, "bogie_f": 0, "bogie_r": 0,
               "nexus_w1": math.ceil(d["w1"]), "nexus_w2": math.ceil(d["w2"]),
               "nexus_w3": math.ceil(d["w3"]), "nexus_w4": 0}

    g_acc       = 9.81
    car_scale   = float(vtcm_ctx["Mc"]) * g_acc
    bogie_scale = float(vtcm_ctx["Mt"]) * g_acc
    _std        = out_norm_stats.get("std", None)
    geom_scale  = (float(np.mean(_std[4:8])) + 1e-6) if _std is not None \
                  else (float(phys[:, 4:8, :].std().item()) + 1e-6)

    def _eq_loss(key: str, scale: float) -> torch.Tensor:
        r = res[key][:, :max(T_seq - pad_d[key], 1)] / scale
        return F.mse_loss(torch.clamp(r, -1000., 1000.),
                          torch.zeros_like(r))

    lc  = _eq_loss("car_body",  car_scale)
    lbf = _eq_loss("bogie_f",   bogie_scale)
    lbr = _eq_loss("bogie_r",   bogie_scale)
    ln1 = _eq_loss("nexus_w1",  geom_scale)
    ln2 = _eq_loss("nexus_w2",  geom_scale)
    ln3 = _eq_loss("nexus_w3",  geom_scale)
    ln4 = _eq_loss("nexus_w4",  geom_scale)

    loss_cb  = (lc + lbf + lbr) / 3.0
    loss_nex = (ln1 + ln2 + ln3 + ln4) / 4.0
    loss_total = torch.clamp(loss_cb + nexus_weight * loss_nex, max=100.0)

    components = {
        "car_body":    lc.detach(),   "bogie_front": lbf.detach(),
        "bogie_rear":  lbr.detach(),  "nexus_w1":    ln1.detach(),
        "nexus_w2":    ln2.detach(),  "nexus_w3":    ln3.detach(),
        "nexus_w4":    ln4.detach(),
    }
    return loss_total, components
