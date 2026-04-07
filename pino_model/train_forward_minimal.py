from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys

# Reduce OpenMP oversubscription by default.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dataset_generator import WindowConfig, build_dataloader
from pino_model.pino_architecture import PINOConfig, PINOResidualHead
from pino_model.dataset_gen import (
    prepare_dataset,
    _parse_components,
    _target_indices,
    _load_run_metadata,
)
from configs.parameters import Antiyawer_parameters, Fastener_KV, IntegrationParams, VehicleParams

torch = None
plt = None


def _require_torch():
    try:
        torch_mod = importlib.import_module("torch")
        F_mod = importlib.import_module("torch.nn.functional")

        return torch_mod, F_mod
    except Exception as e:
        raise RuntimeError("This training script requires torch. Please install PyTorch first.") from e


def _require_matplotlib():
    try:
        return importlib.import_module("matplotlib.pyplot")
    except Exception as e:
        raise RuntimeError("This script requires matplotlib for plotting.") from e





@dataclass
class DynamicsLossContext:
    enabled: bool
    reason: str
    vehicle_params: Optional[VehicleParams] = None
    antiyawer_params: Optional[Antiyawer_parameters] = None
    mass_vehicle: Optional[np.ndarray] = None
    omega: float = 0.0
    reliable_dof_mask: Optional[np.ndarray] = None





def _build_dynamics_context(args) -> DynamicsLossContext:
    if args.lambda_dyn <= 0.0:
        return DynamicsLossContext(enabled=False, reason="`lambda_dyn<=0`，跳过动力学残差。")
    if args.target_object.lower() not in {"vehicle_full", "full_vehicle", "vehicle35"}:
        return DynamicsLossContext(enabled=False, reason="动力学残差要求 `target_object=vehicle_full`（完整35自由度车辆状态）。")
    comps = set(_parse_components(args.components))
    if not {"disp", "vel", "acc"}.issubset(comps):
        return DynamicsLossContext(enabled=False, reason="动力学残差要求同时提供 `disp,vel,acc`。")

    npz_meta = _load_run_metadata(Path(args.npz_path)) if args.npz_path else {}
    vehicle_type = args.vehicle_type or npz_meta.get("vehicle_type", "高速客车")
    param_profile_dir = args.param_profile_dir or npz_meta.get("param_profile_dir", "configs/standard")
    vx_set = args.vx_set if args.vx_set > 0 else float(npz_meta.get("vx_set", 215.0))
    fastener_type = args.fastener_type or npz_meta.get("fastener_type", "Standard_KV")
    g = args.g if args.g > 0 else float(npz_meta.get("g", 9.81))

    vehicle_params = VehicleParams(vehicle_type=vehicle_type, yaml_dir=param_profile_dir)
    if vehicle_params.category != "客车":
        return DynamicsLossContext(enabled=False, reason=f"当前仅为客车公式实现动力学残差，检测到 `{vehicle_params.category}`。")

    antiyawer_params = Antiyawer_parameters(yaml_dir=param_profile_dir)
    fastener = Fastener_KV(fastener_type=fastener_type, yaml_dir=param_profile_dir)
    integration = IntegrationParams(
        Lc=vehicle_params.Lc,
        Lt=vehicle_params.Lt,
        R=vehicle_params.R,
        Lkj=fastener.Lkj,
        Vx_set=vx_set,
        Tz=max(args.dt, 1.0),
        Tstep=args.dt,
        g=g,
    )
    mass_vehicle = np.array([
        vehicle_params.Mc, vehicle_params.Mc, vehicle_params.Jcx, vehicle_params.Jcy, vehicle_params.Jcz,
        vehicle_params.Mt, vehicle_params.Mt, vehicle_params.Jtx, vehicle_params.Jty, vehicle_params.Jtz,
        vehicle_params.Mt, vehicle_params.Mt, vehicle_params.Jtx, vehicle_params.Jty, vehicle_params.Jtz,
        vehicle_params.Mw, vehicle_params.Mw, vehicle_params.Jwx, vehicle_params.Jwy, vehicle_params.Jwz,
        vehicle_params.Mw, vehicle_params.Mw, vehicle_params.Jwx, vehicle_params.Jwy, vehicle_params.Jwz,
        vehicle_params.Mw, vehicle_params.Mw, vehicle_params.Jwx, vehicle_params.Jwy, vehicle_params.Jwz,
        vehicle_params.Mw, vehicle_params.Mw, vehicle_params.Jwx, vehicle_params.Jwy, vehicle_params.Jwz,
    ], dtype=np.float32)
    reliable_mask = np.ones(35, dtype=bool)
    reliable_mask[[17, 18, 19, 22, 23, 24, 27, 28, 29, 32, 33, 34]] = False
    return DynamicsLossContext(
        enabled=True,
        reason="已启用基于 `generate_main.py` 车辆动力学装配的残差损失。",
        vehicle_params=vehicle_params,
        antiyawer_params=antiyawer_params,
        mass_vehicle=mass_vehicle,
        omega=float(integration.omega),
        reliable_dof_mask=reliable_mask,
    )








def _to_torch_batch(batch: Dict[str, np.ndarray], device, torch):
    out = {}
    for k, v in batch.items():
        out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out


def spectral_l1_torch(pred, target, eps: float = 1e-8):
    global torch
    if torch is None:
        raise RuntimeError("torch is not initialized")

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    p = torch.fft.rfft(pred, dim=1)
    t = torch.fft.rfft(target, dim=1)
    p_psd = (p.real**2 + p.imag**2).mean(dim=-1)
    t_psd = (t.real**2 + t.imag**2).mean(dim=-1)
    return torch.mean(torch.abs(torch.log(p_psd + eps) - torch.log(t_psd + eps)))


def _resolve_speed_ms(context_tensor, args):
    global torch
    if context_tensor is not None and context_tensor.ndim == 2 and context_tensor.shape[1] >= 1:
        return torch.clamp(context_tensor[:, 0:1] / 3.6, min=1e-6)
    if args.vx_set > 0:
        return torch.full((1, 1), float(args.vx_set) / 3.6, dtype=torch.float32)
    return torch.full((1, 1), 215.0 / 3.6, dtype=torch.float32)


def compute_kinematic_loss(z_pred, component_segments, z_sigma, dt, ds, domain, context_tensor, args):
    """
    计算基于运动学与动力学的物理残差 (Physics-Informed Loss)。
    目前主要实现运动学约束：速度必须是位移的导数，加速度必须是速度的导数。
    参数:
        z_pred: 归一化条件下的预测值形状 [B, T, C]
        component_segments: 告诉我们 disp, vel, acc 位于特征的哪些列 (例如 {"disp": (0,2), "vel": (2,4)})
        z_sigma: 用于反归一化的标准差 (因为求导在真实尺度上才有准确的物理意义)
        dt: 离散时间步长 (比如 1e-4)
    """
    global torch
    if torch is None:
        raise RuntimeError("torch is not initialized")
        
    s_disp = component_segments.get("disp")
    s_vel = component_segments.get("vel")
    s_acc = component_segments.get("acc")
    
    # 转换为原本的物理尺度
    # 考虑到性能，也可以在归一化尺度上做近似，但这里为了严格物理，先反归一化缩放
    z_pred_real = z_pred * torch.as_tensor(z_sigma, dtype=torch.float32, device=z_pred.device)
    
    loss_kinematic = 0.0
    
    if domain == "space":
        speed_ms = _resolve_speed_ms(context_tensor, args).to(z_pred.device)
        speed_ms = speed_ms.unsqueeze(1)

        # 空间域链式约束：v = v_x * dx/dS
        if s_disp and s_vel:
            x = z_pred_real[..., s_disp[0]:s_disp[1]]
            v = z_pred_real[..., s_vel[0]:s_vel[1]]
            dx_ds = (x[:, 1:, :] - x[:, :-1, :]) / max(ds, 1e-12)
            v_from_x = speed_ms * dx_ds
            loss_kinematic += torch.mean((v_from_x - v[:, :-1, :])**2)

        # 空间域链式约束：a = v_x * dv/dS
        if s_vel and s_acc:
            v = z_pred_real[..., s_vel[0]:s_vel[1]]
            a = z_pred_real[..., s_acc[0]:s_acc[1]]
            dv_ds = (v[:, 1:, :] - v[:, :-1, :]) / max(ds, 1e-12)
            a_from_v = speed_ms * dv_ds
            loss_kinematic += torch.mean((a_from_v - a[:, :-1, :])**2)
    else:
        # 时间域约束：v = dx/dt, a = dv/dt
        if s_disp and s_vel:
            x = z_pred_real[..., s_disp[0]:s_disp[1]]
            v = z_pred_real[..., s_vel[0]:s_vel[1]]
            dx_dt = (x[:, 1:, :] - x[:, :-1, :]) / dt
            loss_kinematic += torch.mean((dx_dt - v[:, :-1, :])**2)

        if s_vel and s_acc:
            v = z_pred_real[..., s_vel[0]:s_vel[1]]
            a = z_pred_real[..., s_acc[0]:s_acc[1]]
            dv_dt = (v[:, 1:, :] - v[:, :-1, :]) / dt
            loss_kinematic += torch.mean((dv_dt - a[:, :-1, :])**2)

    return loss_kinematic


def _torch_interp1d(x, xp, fp):
    global torch
    x_clamped = torch.clamp(x, min=float(xp[0]), max=float(xp[-1]))
    idx = torch.searchsorted(xp, x_clamped, right=True)
    idx1 = torch.clamp(idx, 1, xp.numel() - 1)
    idx0 = idx1 - 1
    x0 = xp[idx0]
    x1 = xp[idx1]
    y0 = fp[idx0]
    y1 = fp[idx1]
    weight = (x_clamped - x0) / (x1 - x0 + 1e-12)
    return y0 + weight * (y1 - y0)


def _extract_vehicle_state_torch(x_car, v_car):
    return {
        "XCar": x_car,
        "VCar": v_car,
        "X_YW": x_car[..., 15:35:5],
        "X_ZW": x_car[..., 16:35:5],
        "X_RollW": x_car[..., 17:35:5],
        "X_SpinW": x_car[..., 18:35:5],
        "X_YawW": x_car[..., 19:35:5],
        "V_YW": v_car[..., 15:35:5],
        "V_ZW": v_car[..., 16:35:5],
        "V_RollW": v_car[..., 17:35:5],
        "V_SpinW": v_car[..., 18:35:5],
        "V_YawW": v_car[..., 19:35:5],
    }


def _compute_passenger_suspension_forces_torch(state, dyn_ctx: DynamicsLossContext):
    global torch
    p = dyn_ctx.vehicle_params
    ap = dyn_ctx.antiyawer_params
    device = state["XCar"].device
    dtype = state["XCar"].dtype

    XCar = state["XCar"]
    VCar = state["VCar"]
    Yc, Zc, Rollc, Spinc, Yawc = [XCar[..., i] for i in range(5)]
    dYc, dZc, dRollc, dSpinc, dYawc = [VCar[..., i] for i in range(5)]

    Yt = XCar[..., 5:15:5]
    Zt = XCar[..., 6:15:5]
    Rollt = XCar[..., 7:15:5]
    Spint = XCar[..., 8:15:5]
    Yawt = XCar[..., 9:15:5]
    dYt = VCar[..., 5:15:5]
    dZt = VCar[..., 6:15:5]
    dRollt = VCar[..., 7:15:5]
    dSpint = VCar[..., 8:15:5]
    dYawt = VCar[..., 9:15:5]

    bogie_index = torch.as_tensor([0, 0, 1, 1], device=device)
    sign_pow = torch.as_tensor([-1.0, 1.0, -1.0, 1.0], device=device, dtype=dtype).view(1, 1, 4)
    sign_pow_sec = torch.as_tensor([-1.0, 1.0], device=device, dtype=dtype).view(1, 1, 2)

    Yt_w = Yt.index_select(-1, bogie_index)
    Zt_w = Zt.index_select(-1, bogie_index)
    Rollt_w = Rollt.index_select(-1, bogie_index)
    Spint_w = Spint.index_select(-1, bogie_index)
    Yawt_w = Yawt.index_select(-1, bogie_index)
    dYt_w = dYt.index_select(-1, bogie_index)
    dZt_w = dZt.index_select(-1, bogie_index)
    dRollt_w = dRollt.index_select(-1, bogie_index)
    dSpint_w = dSpint.index_select(-1, bogie_index)
    dYawt_w = dYawt.index_select(-1, bogie_index)

    static_force_pre = torch.as_tensor((p.Mc + 2 * p.Mt) * 9.81 / 8.0, device=device, dtype=dtype)
    static_force_sec = torch.as_tensor(p.Mc * 9.81 / 4.0, device=device, dtype=dtype)

    Fxf_L = p.Kpx * (p.dw * Yawt_w + p.Htw * Spint_w - p.dw * state["X_YawW"]) + p.Cpx * (p.dw * dYawt_w + p.Htw * dSpint_w - p.dw * state["V_YawW"])
    Fxf_R = p.Kpx * (-p.dw * Yawt_w + p.Htw * Spint_w + p.dw * state["X_YawW"]) + p.Cpx * (-p.dw * dYawt_w + p.Htw * dSpint_w + p.dw * state["V_YawW"])

    Fyf_L = p.Kpy * (state["X_YW"] - Yt_w + p.Htw * Rollt_w - sign_pow * p.Lt * Yawt_w) + p.Cpy * (state["V_YW"] - dYt_w + p.Htw * dRollt_w - sign_pow * p.Lt * dYawt_w)
    Fyf_R = Fyf_L

    Fzf_L = p.Kpz * (Zt_w - state["X_ZW"] - sign_pow * p.Lt * Spint_w + p.dw * state["X_RollW"] - p.dw * Rollt_w) + p.Cpz * (dZt_w - state["V_ZW"] - sign_pow * p.Lt * dSpint_w + p.dw * state["V_RollW"] - p.dw * dRollt_w) + static_force_pre
    Fzf_R = p.Kpz * (Zt_w - state["X_ZW"] - sign_pow * p.Lt * Spint_w - p.dw * state["X_RollW"] + p.dw * Rollt_w) + p.Cpz * (dZt_w - state["V_ZW"] - sign_pow * p.Lt * dSpint_w - p.dw * state["V_RollW"] + p.dw * dRollt_w) + static_force_pre

    Fxt_L = p.Ksx * (p.HcB * Spinc.unsqueeze(-1) + p.HBt * Spint + p.ds * Yawc.unsqueeze(-1) - p.ds * Yawt) + p.Csx * (p.HcB * dSpinc.unsqueeze(-1) + p.HBt * dSpint + p.ds * dYawc.unsqueeze(-1) - p.ds * dYawt)
    Fxt_R = p.Ksx * (p.HcB * Spinc.unsqueeze(-1) + p.HBt * Spint - p.ds * Yawc.unsqueeze(-1) + p.ds * Yawt) + p.Csx * (p.HcB * dSpinc.unsqueeze(-1) + p.HBt * dSpint - p.ds * dYawc.unsqueeze(-1) + p.ds * dYawt)

    Fyt_L = p.Ksy * (Yt - Yc.unsqueeze(-1) + p.HBt * Rollt + p.HcB * Rollc.unsqueeze(-1) + sign_pow_sec * p.Lc * Yawc.unsqueeze(-1)) + p.Csy * (dYt - dYc.unsqueeze(-1) + p.HBt * dRollt + p.HcB * dRollc.unsqueeze(-1) + sign_pow_sec * p.Lc * dYawc.unsqueeze(-1))
    Fyt_R = Fyt_L

    Fzt_L = p.Ksz * (Zc.unsqueeze(-1) - Zt + p.ds * Rollt - p.ds * Rollc.unsqueeze(-1) + sign_pow_sec * p.Lc * Spinc.unsqueeze(-1)) + p.Csz * (dZc.unsqueeze(-1) - dZt + p.ds * dRollt - p.ds * dRollc.unsqueeze(-1) + sign_pow_sec * p.Lc * dSpinc.unsqueeze(-1)) + static_force_sec
    Fzt_R = p.Ksz * (Zc.unsqueeze(-1) - Zt - p.ds * Rollt + p.ds * Rollc.unsqueeze(-1) + sign_pow_sec * p.Lc * Spinc.unsqueeze(-1)) + p.Csz * (dZc.unsqueeze(-1) - dZt - p.ds * dRollt + p.ds * dRollc.unsqueeze(-1) + sign_pow_sec * p.Lc * dSpinc.unsqueeze(-1)) + static_force_sec

    Mr = p.Krx * (Rollc.unsqueeze(-1) - Rollt)

    yaw_damper_v = torch.as_tensor(ap.yaw_damper_v, device=device, dtype=dtype)
    yaw_damper_f = torch.as_tensor(ap.yaw_damper_f, device=device, dtype=dtype)
    vxct_L = ap.dsc * dYawc.unsqueeze(-1) - ap.dsc * dYawt + p.HcB * dSpinc.unsqueeze(-1) + p.HBt * dSpint
    vxct_R = -ap.dsc * dYawc.unsqueeze(-1) + ap.dsc * dYawt + p.HcB * dSpinc.unsqueeze(-1) + p.HBt * dSpint
    xxct_L = ap.dsc * Yawc.unsqueeze(-1) - ap.dsc * Yawt + p.HcB * Spinc.unsqueeze(-1) + p.HBt * Spint
    xxct_R = -ap.dsc * Yawc.unsqueeze(-1) + ap.dsc * Yawt + p.HcB * Spinc.unsqueeze(-1) + p.HBt * Spint
    Fxs_L = _torch_interp1d(vxct_L, yaw_damper_v, yaw_damper_f) + ap.kantiyawer * xxct_L
    Fxs_R = _torch_interp1d(vxct_R, yaw_damper_v, yaw_damper_f) + ap.kantiyawer * xxct_R

    return {
        "Fxf_L": Fxf_L, "Fxf_R": Fxf_R,
        "Fyf_L": Fyf_L, "Fyf_R": Fyf_R,
        "Fzf_L": Fzf_L, "Fzf_R": Fzf_R,
        "Fxt_L": Fxt_L, "Fxt_R": Fxt_R,
        "Fyt_L": Fyt_L, "Fyt_R": Fyt_R,
        "Fzt_L": Fzt_L, "Fzt_R": Fzt_R,
        "Mr": Mr, "Fxs_L": Fxs_L, "Fxs_R": Fxs_R,
    }


def _assemble_vehicle_gf_torch(state, susp_forces, wr_vertical, wr_lateral, wr_vertical_p2, wr_lateral_p2, dyn_ctx: DynamicsLossContext):
    global torch
    p = dyn_ctx.vehicle_params
    ap = dyn_ctx.antiyawer_params
    device = state["XCar"].device
    dtype = state["XCar"].dtype

    zeros4 = torch.zeros_like(wr_vertical[..., :4])
    TFx_L, TFx_R = zeros4, zeros4
    TFy_L, TFy_R = wr_lateral[..., :4], wr_lateral[..., 4:]
    TFz_L, TFz_R = wr_vertical[..., :4], wr_vertical[..., 4:]
    TFx_L2, TFx_R2 = zeros4, zeros4
    TFy_L2 = wr_lateral_p2[..., :4] if wr_lateral_p2 is not None else zeros4
    TFy_R2 = wr_lateral_p2[..., 4:] if wr_lateral_p2 is not None else zeros4
    TFz_L2 = wr_vertical_p2[..., :4] if wr_vertical_p2 is not None else zeros4
    TFz_R2 = wr_vertical_p2[..., 4:] if wr_vertical_p2 is not None else zeros4
    MLy = torch.zeros_like(TFy_L)
    MLz = torch.zeros_like(TFy_L)
    MRy = torch.zeros_like(TFy_L)
    MRz = torch.zeros_like(TFy_L)
    rL = torch.zeros_like(TFy_L)
    rR = torch.zeros_like(TFy_L)
    rL2 = torch.zeros_like(TFy_L)
    rR2 = torch.zeros_like(TFy_L)
    a0 = torch.zeros_like(TFy_L)
    a02 = torch.zeros_like(TFy_L)

    Fxf_L, Fxf_R = susp_forces["Fxf_L"], susp_forces["Fxf_R"]
    Fyf_L, Fyf_R = susp_forces["Fyf_L"], susp_forces["Fyf_R"]
    Fzf_L, Fzf_R = susp_forces["Fzf_L"], susp_forces["Fzf_R"]
    Fxt_L, Fxt_R = susp_forces["Fxt_L"], susp_forces["Fxt_R"]
    Fyt_L, Fyt_R = susp_forces["Fyt_L"], susp_forces["Fyt_R"]
    Fzt_L, Fzt_R = susp_forces["Fzt_L"], susp_forces["Fzt_R"]
    Mr = susp_forces["Mr"]
    Fxs_L, Fxs_R = susp_forces["Fxs_L"], susp_forces["Fxs_R"]

    VWRoll = state["V_RollW"]
    VWSpin = state["V_SpinW"]
    VWYaw = state["V_YawW"]
    XWYaw = state["X_YawW"]
    omega = torch.as_tensor(dyn_ctx.omega, device=device, dtype=dtype)
    gravity = torch.as_tensor(9.81, device=device, dtype=dtype)

    GF_Wheelset_Y = -Fyf_L - Fyf_R + TFy_L + TFy_R + TFy_L2 + TFy_R2
    GF_Wheelset_Z = -TFz_L - TFz_R + Fzf_L + Fzf_R + p.Mw * gravity - TFz_L2 - TFz_R2
    GF_Wheelset_Roll = p.Jwy * (VWSpin - omega) * VWYaw + a0 * (TFz_L - TFz_R) - rL * TFy_L - rR * TFy_R + p.dw * (Fzf_R - Fzf_L) + a02 * (TFz_L2 - TFz_R2) - rL2 * TFy_L2 - rR2 * TFy_R2
    GF_Wheelset_Yaw = p.Jwy * VWRoll * (VWSpin - omega) + a0 * (TFx_L - TFx_R) + a0 * XWYaw * (TFy_L - TFy_R) + MLz + MRz + p.dw * (Fxf_L - Fxf_R) + a02 * (TFx_L2 - TFx_R2) + a02 * XWYaw * (TFy_L2 - TFy_R2)
    GF_Wheelset_Spin = rR * TFx_R + rL * TFx_L + rR * XWYaw * TFy_R + rL * XWYaw * TFy_L + MLy + MRy + rR2 * TFx_R2 + rL2 * TFx_L2 + rR2 * XWYaw * TFy_R2 + rL2 * XWYaw * TFy_L2

    idx1 = torch.as_tensor([0, 2], device=device)
    idx2 = torch.as_tensor([1, 3], device=device)
    GF_Bogie_Y = Fyf_L.index_select(-1, idx1) + Fyf_L.index_select(-1, idx2) - Fyt_L + Fyf_R.index_select(-1, idx1) + Fyf_R.index_select(-1, idx2) - Fyt_R
    GF_Bogie_Z = Fzt_L - Fzf_L.index_select(-1, idx1) - Fzf_L.index_select(-1, idx2) + Fzt_R - Fzf_R.index_select(-1, idx1) - Fzf_R.index_select(-1, idx2) + p.Mt * gravity
    GF_Bogie_Roll = -(Fyf_L.index_select(-1, idx1) + Fyf_R.index_select(-1, idx1) + Fyf_L.index_select(-1, idx2) + Fyf_R.index_select(-1, idx2)) * p.Htw + (Fzf_L.index_select(-1, idx1) + Fzf_L.index_select(-1, idx2) - Fzf_R.index_select(-1, idx1) - Fzf_R.index_select(-1, idx2)) * p.dw + (Fzt_R - Fzt_L) * p.ds - (Fyt_L + Fyt_R) * p.HBt + Mr
    GF_Bogie_Yaw = (Fyf_L.index_select(-1, idx1) + Fyf_R.index_select(-1, idx1) - Fyf_L.index_select(-1, idx2) - Fyf_R.index_select(-1, idx2)) * p.Lt + (Fxf_R.index_select(-1, idx1) + Fxf_R.index_select(-1, idx2) - Fxf_L.index_select(-1, idx1) - Fxf_L.index_select(-1, idx2)) * p.dw + (Fxt_L - Fxt_R) * p.ds + (Fxs_L - Fxs_R) * ap.dsc
    GF_Bogie_Spin = (Fzf_L.index_select(-1, idx1) + Fzf_R.index_select(-1, idx1) - Fzf_L.index_select(-1, idx2) - Fzf_R.index_select(-1, idx2)) * p.Lt - (Fxf_L.index_select(-1, idx1) + Fxf_R.index_select(-1, idx1) + Fxf_L.index_select(-1, idx2) + Fxf_R.index_select(-1, idx2)) * p.Htw - (Fxt_L + Fxt_R) * p.HBt - (Fxs_L + Fxs_R) * p.HBt

    GF_Car_Y = torch.sum(Fyt_L, dim=-1) + torch.sum(Fyt_R, dim=-1)
    GF_Car_Z = -torch.sum(Fzt_L, dim=-1) - torch.sum(Fzt_R, dim=-1) + p.Mc * gravity
    GF_Car_Roll = -torch.sum(Fyt_L + Fyt_R, dim=-1) * p.HcB + (Fzt_L[..., 0] + Fzt_L[..., 1] - Fzt_R[..., 0] - Fzt_R[..., 1]) * p.ds - torch.sum(Mr, dim=-1)
    GF_Car_Spin = (Fzt_L[..., 0] + Fzt_R[..., 0] - Fzt_L[..., 1] - Fzt_R[..., 1]) * p.Lc - torch.sum(Fxt_L + Fxt_R, dim=-1) * p.HcB - torch.sum(Fxs_L + Fxs_R, dim=-1) * p.HcB
    GF_Car_Yaw = (Fyt_L[..., 0] + Fyt_R[..., 0] - Fyt_L[..., 1] - Fyt_R[..., 1]) * p.Lc + torch.sum(Fxt_R, dim=-1) * p.ds - torch.sum(Fxt_L, dim=-1) * p.ds + torch.sum(Fxs_R, dim=-1) * ap.dsc - torch.sum(Fxs_L, dim=-1) * ap.dsc

    return torch.cat([
        torch.stack([GF_Car_Y, GF_Car_Z, GF_Car_Roll, GF_Car_Spin, GF_Car_Yaw], dim=-1),
        torch.stack([GF_Bogie_Y[..., 0], GF_Bogie_Z[..., 0], GF_Bogie_Roll[..., 0], GF_Bogie_Spin[..., 0], GF_Bogie_Yaw[..., 0]], dim=-1),
        torch.stack([GF_Bogie_Y[..., 1], GF_Bogie_Z[..., 1], GF_Bogie_Roll[..., 1], GF_Bogie_Spin[..., 1], GF_Bogie_Yaw[..., 1]], dim=-1),
        torch.stack([GF_Wheelset_Y[..., 0], GF_Wheelset_Z[..., 0], GF_Wheelset_Roll[..., 0], GF_Wheelset_Spin[..., 0], GF_Wheelset_Yaw[..., 0]], dim=-1),
        torch.stack([GF_Wheelset_Y[..., 1], GF_Wheelset_Z[..., 1], GF_Wheelset_Roll[..., 1], GF_Wheelset_Spin[..., 1], GF_Wheelset_Yaw[..., 1]], dim=-1),
        torch.stack([GF_Wheelset_Y[..., 2], GF_Wheelset_Z[..., 2], GF_Wheelset_Roll[..., 2], GF_Wheelset_Spin[..., 2], GF_Wheelset_Yaw[..., 2]], dim=-1),
        torch.stack([GF_Wheelset_Y[..., 3], GF_Wheelset_Z[..., 3], GF_Wheelset_Roll[..., 3], GF_Wheelset_Spin[..., 3], GF_Wheelset_Yaw[..., 3]], dim=-1),
    ], dim=-1)


def compute_dynamics_loss(z_pred, component_segments, z_sigma, batch, dynamics_context: Optional[DynamicsLossContext]):
    global torch
    if torch is None:
        raise RuntimeError("torch is not initialized")
    if dynamics_context is None or not dynamics_context.enabled:
        return z_pred.new_tensor(0.0)
    if batch is None:
        return z_pred.new_tensor(0.0)
    if "wr_force_vertical" not in batch or "wr_force_lateral" not in batch:
        return z_pred.new_tensor(0.0)

    s_disp = component_segments.get("disp")
    s_vel = component_segments.get("vel")
    s_acc = component_segments.get("acc")
    if not (s_disp and s_vel and s_acc):
        return z_pred.new_tensor(0.0)

    z_sigma_t = torch.as_tensor(z_sigma, dtype=torch.float32, device=z_pred.device)
    z_pred_real = z_pred * z_sigma_t
    x_car = z_pred_real[..., s_disp[0]:s_disp[1]]
    v_car = z_pred_real[..., s_vel[0]:s_vel[1]]
    a_pred = z_pred_real[..., s_acc[0]:s_acc[1]]
    if x_car.shape[-1] != 35 or v_car.shape[-1] != 35 or a_pred.shape[-1] != 35:
        return z_pred.new_tensor(0.0)

    state = _extract_vehicle_state_torch(x_car, v_car)
    susp_forces = _compute_passenger_suspension_forces_torch(state, dynamics_context)
    wr_vertical = batch["wr_force_vertical"]
    wr_lateral = batch["wr_force_lateral"]
    wr_vertical_p2 = batch.get("wr_force_vertical_p2")
    wr_lateral_p2 = batch.get("wr_force_lateral_p2")
    gf_vehicle = _assemble_vehicle_gf_torch(state, susp_forces, wr_vertical, wr_lateral, wr_vertical_p2, wr_lateral_p2, dynamics_context)

    mass_vehicle = torch.as_tensor(dynamics_context.mass_vehicle, dtype=torch.float32, device=z_pred.device)
    reliable_mask = torch.as_tensor(dynamics_context.reliable_dof_mask, dtype=torch.bool, device=z_pred.device)
    a_dyn = gf_vehicle[..., reliable_mask] / mass_vehicle[reliable_mask]
    return torch.mean((a_pred[..., reliable_mask] - a_dyn) ** 2)


def build_input_features(z_in, u_in=None, context=None):
    global torch
    if torch is None:
        raise RuntimeError("torch is not initialized")

    feats = [z_in]
    if u_in is not None:
        feats.append(u_in)
    if context is not None:
        ctx = context.unsqueeze(1).expand(-1, z_in.size(1), -1)
        feats.append(ctx)
    return torch.cat(feats, dim=-1)


def evaluate_epoch(model, loader, device, F, args, component_segments, z_sigma, dynamics_context):
    global torch
    model.eval()
    total = 0.0
    n = 0
    sample_pack = None

    with torch.no_grad():
        for batch_np in loader:
            batch = _to_torch_batch(batch_np, device, torch)
            z_in = batch["z_in"]
            z_tgt = batch["z_tgt"]
            u_in = batch.get("u_in")
            context = batch.get("context")

            x = build_input_features(z_in, u_in=u_in, context=context)
            dz = model(x)
            z_pred = z_in + dz

            loss_time = F.mse_loss(z_pred, z_tgt)
            loss_spec = spectral_l1_torch(z_pred, z_tgt)
            loss_kin = compute_kinematic_loss(
                z_pred=z_pred,
                component_segments=component_segments,
                z_sigma=z_sigma,
                dt=args.dt,
                ds=args.ds,
                domain=args.domain,
                context_tensor=context,
                args=args,
            )
            loss_dyn = compute_dynamics_loss(z_pred, component_segments, z_sigma, batch, dynamics_context)
            
            loss = args.lambda_time * loss_time + args.lambda_spec * loss_spec + args.lambda_phy * loss_kin + args.lambda_dyn * loss_dyn
            total += float(loss.detach().cpu())
            n += 1

            if sample_pack is None:
                sample_pack = {
                    "pred": z_pred[0].detach().cpu().numpy(),
                    "true": z_tgt[0].detach().cpu().numpy(),
                }

    return (total / max(n, 1)), sample_pack


def evaluate_epoch_full_sequence(model, test_records, device, F, args, z_mu, z_sigma):
    """
    Evaluate model on a full test window from the first test record.
    Used for full-segment visualization during training.
    
    Returns normalized predictions and ground truth for a complete window.
    """
    global torch
    if torch is None:
        raise RuntimeError("torch is not initialized")
    
    model.eval()
    
    # Use only the first test record for visualization
    if not test_records:
        return None
    
    rec = test_records[0]
    
    with torch.no_grad():
        z_full = torch.as_tensor(rec["z"], dtype=torch.float32, device=device).unsqueeze(0)
        
        if "u" in rec:
            u_full = torch.as_tensor(rec["u"], dtype=torch.float32, device=device).unsqueeze(0)
        else:
            u_full = None
        
        if "c" in rec:
            c_full = torch.as_tensor(rec["c"], dtype=torch.float32, device=device).unsqueeze(0)
        else:
            c_full = None
        
        t_max = z_full.shape[1]
        
        # Extract a window starting from index 0
        # z_in: [1, window_size, features], z_tgt: [1, window_size, features]
        if t_max < 2 * args.window_size:
            # Sequence too short
            return None
        
        z_in_start = 0
        z_in_end = args.window_size
        z_tgt_start = args.window_size
        z_tgt_end = 2 * args.window_size
        
        z_in = z_full[:, z_in_start:z_in_end, :]
        
        if u_full is not None:
            u_in = u_full[:, z_in_start:z_in_end, :]
        else:
            u_in = None
        
        x = build_input_features(z_in, u_in=u_in, context=c_full)
        dz = model(x)
        z_pred = z_in + dz
        z_tgt = z_full[:, z_tgt_start:z_tgt_end, :]
        
        # Return the predicted window and the target window (normalized)
        pred_all = z_pred[0].detach().cpu().numpy()
        true_all = z_tgt[0].detach().cpu().numpy()
        
        return {
            "pred": pred_all,
            "true": true_all,
        }
    
    return None


def plot_loss_curves(train_losses: List[float], test_losses: List[float], save_path: Path) -> None:
    global plt
    if plt is None:
        plt = _require_matplotlib()

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    ax.plot(epochs, train_losses, label="Train Loss", lw=1.8)
    ax.plot(epochs, test_losses, label="Test Loss", lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training/Test Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_pred_vs_true(
    sample_pack: Dict[str, np.ndarray],
    component_segments: Dict[str, Tuple[int, int]],
    channel_id: int,
    save_path: Path,
    z_mu: np.ndarray,
    z_sigma: np.ndarray,
):
    global plt
    if plt is None:
        plt = _require_matplotlib()

    pred = sample_pack["pred"] * z_sigma + z_mu
    true = sample_pack["true"] * z_sigma + z_mu
    t = np.arange(pred.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=130, sharex=True)
    comp_order = ["disp", "vel", "acc"]
    labels = {"disp": "Displacement", "vel": "Velocity", "acc": "Acceleration"}

    for i, c in enumerate(comp_order):
        ax = axes[i]
        if c not in component_segments:
            ax.text(0.5, 0.5, f"{labels[c]} not enabled", ha="center", va="center")
            ax.axis("off")
            continue
        s, e = component_segments[c]
        ch = min(channel_id, e - s - 1)
        j = s + ch
        ax.plot(t, true[:, j], label="True", lw=1.5)
        ax.plot(t, pred[:, j], label="Pred", lw=1.2, alpha=0.9)
        ax.set_ylabel(labels[c])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time Step in Window")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def train_forward(args):
    global torch, plt

    if args.allow_dup_omp:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    torch, F = _require_torch()
    plt = _require_matplotlib()

    # Prepare dataset
    dataset_dir = Path("Dataset")
    u_keys = [k.strip() for k in args.u_keys.split(",") if k.strip()]
    
    train_loader, test_loader, component_segments, selected_indices, z_mu, z_sigma, ds, test_records = prepare_dataset(
        npz_path=Path(args.npz_path) if not args.demo else None,
        target_object=args.target_object,
        components=args.components,
        window_size=args.window_size,
        stride=args.stride,
        pred_horizon=args.pred_horizon,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        u_keys=u_keys,
        demo=args.demo,
        demo_records=args.demo_records,
        demo_length=args.demo_length,
        dataset_dir=dataset_dir,
    )
    
    # Update args.ds if auto-detected
    if ds > 0:
        args.ds = ds
        if not args.demo and args.domain == "space":
            print(f"[Domain] 空间域模式已启用，自动识别 ds={args.ds:.6f} m")

    dynamics_context = _build_dynamics_context(args)
    print(f"[Dynamics Loss] {dynamics_context.reason}")

    first = next(iter(train_loader))
    cin = first["z_in"].shape[-1] + first.get("u_in", np.zeros((1, 1, 0))).shape[-1] + first.get("context", np.zeros((1, 0))).shape[-1]
    cout = first["z_in"].shape[-1]

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PINOResidualHead(PINOConfig(in_channels=cin, out_channels=cout, width=args.width, modes=args.modes, n_layers=args.layers)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_hist: List[float] = []
    test_loss_hist: List[float] = []
    vis_dir = Path(args.output).with_suffix("").parent / "figures"

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_step = 0

        for batch_np in train_loader:
            batch = _to_torch_batch(batch_np, device, torch)
            z_in = batch["z_in"]
            z_tgt = batch["z_tgt"]
            u_in = batch.get("u_in")
            context = batch.get("context")

            x = build_input_features(z_in, u_in=u_in, context=context)
            dz = model(x)
            z_pred = z_in + dz

            loss_time = F.mse_loss(z_pred, z_tgt)
            loss_spec = spectral_l1_torch(z_pred, z_tgt)
            loss_kin = compute_kinematic_loss(
                z_pred=z_pred,
                component_segments=component_segments,
                z_sigma=z_sigma,
                dt=args.dt,
                ds=args.ds,
                domain=args.domain,
                context_tensor=context,
                args=args,
            )
            loss_dyn = compute_dynamics_loss(z_pred, component_segments, z_sigma, batch, dynamics_context)
            loss = args.lambda_time * loss_time + args.lambda_spec * loss_spec + args.lambda_phy * loss_kin + args.lambda_dyn * loss_dyn

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += float(loss.detach().cpu())
            n_step += 1

        train_avg = total_loss / max(n_step, 1)
        test_avg, sample = evaluate_epoch(model, test_loader, device, F, args, component_segments, z_sigma, dynamics_context)
        train_loss_hist.append(train_avg)
        test_loss_hist.append(test_avg)

        print(f"Epoch {ep:03d} | train_loss={train_avg:.6f} | test_loss={test_avg:.6f}")

        if ep % args.eval_interval == 0 and sample is not None:
            # Evaluate on full test sequences for better visualization
            full_sample = evaluate_epoch_full_sequence(model, test_records, device, F, args, z_mu, z_sigma)
            if full_sample is not None:
                plot_pred_vs_true(
                    sample_pack=full_sample,
                    component_segments=component_segments,
                    channel_id=args.plot_channel,
                    save_path=vis_dir / f"pred_vs_true_epoch_{ep:03d}.png",
                    z_mu=z_mu,
                    z_sigma=z_sigma,
                )

    plot_loss_curves(
        train_losses=train_loss_hist,
        test_losses=test_loss_hist,
        save_path=vis_dir / "loss_curve_train_test.png",
    )

    ckpt = {
        "model_state": model.state_dict(),
        "config": vars(args),
        "z_mu": z_mu,
        "z_sigma": z_sigma,
        "component_segments": component_segments,
        "selected_vehicle_indices": selected_indices,
        "train_loss": train_loss_hist,
        "test_loss": test_loss_hist,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"Saved checkpoint: {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Minimal forward training for PINO residual model")

    p.add_argument("--demo", action="store_true", help="Use generated demo data")
    p.add_argument("--demo-records", type=int, default=16)
    p.add_argument("--demo-length", type=int, default=2000)
    p.add_argument("--allow-dup-omp", action="store_true", help="Unsafe workaround for OpenMP duplicate runtime conflict")

    p.add_argument("--npz-path", type=str, default="results/default_project/高速客车-外部导入-vehicle-standard-20260322_065703/files/simulation_result_spatial.npz", help="Path to simulation result npz when not using --demo")
    p.add_argument("--u-keys", type=str, default="Irre_bz_L_ref,Irre_bz_R_ref,Irre_by_L_ref,Irre_by_R_ref,Irre_bz_left_ref,Irre_bz_right_ref,Irre_by_left_ref,Irre_by_right_ref")
    p.add_argument("--target-object", type=str, default="vehicle_full", choices=["carbody", "bogie", "wheelset", "all_vehicle", "vehicle_full", "full_vehicle", "vehicle35"])
    p.add_argument("--components", type=str, default="disp,vel,acc", help="Comma-separated: disp,vel,acc")
    p.add_argument("--plot-channel", type=int, default=0, help="Local channel id in selected object for comparison plots")

    p.add_argument("--window-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--pred-horizon", type=int, default=1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--eval-interval", type=int, default=10)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--width", type=int, default=64)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--layers", type=int, default=4)

    p.add_argument("--lambda-time", type=float, default=1.0)
    p.add_argument("--lambda-spec", type=float, default=0.1)
    p.add_argument("--lambda-phy", type=float, default=0.01, help="物理运动学/动力学损失权重")
    p.add_argument("--lambda-dyn", type=float, default=0.01, help="`generate_main.py` 车辆动力学残差权重")
    p.add_argument("--domain", type=str, default="space", choices=["time", "space"], help="训练域：time=时间域，space=空间域")
    p.add_argument("--dt", type=float, default=1e-4, help="系统离散时间步长(s)，用于求导/物理Loss")
    p.add_argument("--ds", type=float, default=0.25, help="空间域离散步长(m)，仅在 --domain space 生效")
    p.add_argument("--vehicle-type", type=str, default="", help="覆盖运行元数据中的车辆类型")
    p.add_argument("--param-profile-dir", type=str, default="", help="覆盖运行元数据中的参数目录")
    p.add_argument("--fastener-type", type=str, default="", help="覆盖运行元数据中的扣件类型")
    p.add_argument("--vx-set", type=float, default=-1.0, help="覆盖运行元数据中的运行速度(km/h)")
    p.add_argument("--g", type=float, default=-1.0, help="覆盖运行元数据中的重力加速度")

    p.add_argument("--device", type=str, default="")
    p.add_argument("--output", type=str, default="results/lnn_pino/forward_minimal.ckpt")

    args = p.parse_args()
    if not args.demo and not args.npz_path:
        p.error("When not using --demo, you must provide --npz-path")
    return args


if __name__ == "__main__":
    train_forward(parse_args())
