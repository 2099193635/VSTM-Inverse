from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.parameters import (
    VehicleParams,
    Fastener_KV,
    IntegrationParams,
    Antiyawer_parameters,
    RailParams,
    Subrail_Params,
    ModesParameters,
    ExtraForceElements_parameters,
)
from physics_modules.suspension import SuspensionSystem
from physics_modules.equation_of_motion import GeneralForceAssembler
import pino_model.train_forward_minimal as tfm


def _load_meta(npz_path: Path) -> dict:
    meta_path = npz_path.parent / "argparse_params.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _build_wr_dict(fy, fz, fy2, fz2):
    zeros4 = np.zeros(4, dtype=np.float64)
    return {
        "FNx_L": zeros4.copy(),
        "FNy_L": fy[:4].copy(),
        "FNz_L": fz[:4].copy(),
        "FNx_R": zeros4.copy(),
        "FNy_R": fy[4:].copy(),
        "FNz_R": fz[4:].copy(),
        "FNx_L2": zeros4.copy(),
        "FNy_L2": fy2[:4].copy(),
        "FNz_L2": fz2[:4].copy(),
        "FNx_R2": zeros4.copy(),
        "FNy_R2": fy2[4:].copy(),
        "FNz_R2": fz2[4:].copy(),
        "MLy": zeros4.copy(),
        "MLz": zeros4.copy(),
        "MRy": zeros4.copy(),
        "MRz": zeros4.copy(),
        "rL": zeros4.copy(),
        "rR": zeros4.copy(),
        "rL2": zeros4.copy(),
        "rR2": zeros4.copy(),
        "a0": zeros4.copy(),
        "a02": zeros4.copy(),
        "CreepForce_L": np.zeros((4, 3), dtype=np.float64),
        "CreepForce_R": np.zeros((4, 3), dtype=np.float64),
        "hrL": zeros4.copy(),
        "eL": zeros4.copy(),
        "hrR": zeros4.copy(),
        "eR": zeros4.copy(),
        "hrL2": zeros4.copy(),
        "eL2": zeros4.copy(),
        "hrR2": zeros4.copy(),
        "eR2": zeros4.copy(),
    }


def main():
    p = argparse.ArgumentParser(description="Validate consistency between generate_main vehicle dynamics and PINO embedded chain")
    p.add_argument("--npz", type=str, default= 'results/default_project/高速客车-外部导入-vehicle-standard-20260322_082429/files/simulation_result_spatial.npz', help="Path to simulation_result.npz or simulation_result_spatial.npz")
    p.add_argument("--samples", type=int, default=25, help="Number of sampled steps")
    p.add_argument("--save-dir", type=str, default="results/lnn_pino/consistency_figs", help="Directory for output figures")
    args = p.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=True)
    meta = _load_meta(npz_path)

    veh_type = meta.get("vehicle_type", "高速客车")
    param_dir = meta.get("param_profile_dir", "configs/standard")
    fast_type = meta.get("fastener_type", "Standard_KV")
    rail_type = meta.get("rail_type", "CHN60")
    vx_set = float(meta.get("vx_set", 215.0))
    g = float(meta.get("g", 9.81))

    veh = VehicleParams(vehicle_type=veh_type, yaml_dir=param_dir)
    fast = Fastener_KV(fastener_type=fast_type, yaml_dir=param_dir)
    integ = IntegrationParams(Lc=veh.Lc, Lt=veh.Lt, R=veh.R, Lkj=fast.Lkj, Vx_set=vx_set, Tz=1.0, Tstep=1e-4, g=g)
    ap = Antiyawer_parameters(yaml_dir=param_dir)
    ep = ExtraForceElements_parameters(Lc=veh.Lc, yaml_dir=param_dir)
    rail = RailParams(rail_type=rail_type, yaml_dir=param_dir)
    sub = Subrail_Params(subrail_type="Standard_Subrail", yaml_dir=param_dir)
    mode = ModesParameters(NV=8, NL=8, NT=8)

    susp = SuspensionSystem(veh_params=veh, antiyawer_params=ap, extra_params=ep)
    assembler = GeneralForceAssembler(
        veh_params=veh,
        integration_params=integ,
        rail_params=rail,
        subrail_params=sub,
        mode_params=mode,
        anitiyawer_params=ap,
    )

    dyn_args = SimpleNamespace(
        lambda_dyn=0.01,
        target_object="vehicle_full",
        components="disp,vel,acc",
        npz_path=str(npz_path),
        vehicle_type=veh_type,
        param_profile_dir=param_dir,
        fastener_type=fast_type,
        vx_set=vx_set,
        g=g,
        dt=1e-4,
    )
    tfm.torch = torch
    dyn_ctx = tfm._build_dynamics_context(dyn_args)

    X = np.asarray(data["X"], dtype=np.float64)
    V = np.asarray(data["V"], dtype=np.float64)
    A = np.asarray(data["A"], dtype=np.float64)
    Fy = np.asarray(data["TotalLateralForce"], dtype=np.float64)
    Fz = np.asarray(data["TotalVerticalForce"], dtype=np.float64)
    Fy2 = np.asarray(data["TotalLateralForce_Point2"], dtype=np.float64) if "TotalLateralForce_Point2" in data else np.zeros_like(Fy)
    Fz2 = np.asarray(data["TotalVerticalForce_Point2"], dtype=np.float64) if "TotalVerticalForce_Point2" in data else np.zeros_like(Fz)

    n = len(X)
    k = max(1, min(args.samples, n - 3))
    idxs = np.linspace(max(2, n // 10), n - 2, num=k, dtype=int)

    gf_absmax, gf_rmse = [], []
    a_absmax, a_rmse = [], []
    a_saved_absmax, a_saved_rmse = [], []

    gf_dim = 35
    a_np_stack = np.zeros((len(idxs), gf_dim), dtype=np.float64)
    a_t_stack = np.zeros((len(idxs), gf_dim), dtype=np.float64)
    a_saved_stack = np.zeros((len(idxs), gf_dim), dtype=np.float64)

    for i in idxs:
        xcar = X[i, :35]
        vcar = V[i, :35]

        state_np = SimpleNamespace(
            XCar=xcar,
            VCar=vcar,
            X_YW=xcar[15:35:5],
            X_ZW=xcar[16:35:5],
            X_RollW=xcar[17:35:5],
            X_SpinW=xcar[18:35:5],
            X_YawW=xcar[19:35:5],
            V_XW=np.full(4, integ.Vc),
            V_YW=vcar[15:35:5],
            V_ZW=vcar[16:35:5],
            V_RollW=vcar[17:35:5],
            V_SpinW=vcar[18:35:5],
            V_YawW=vcar[19:35:5],
        )

        wr_np = _build_wr_dict(Fy[i], Fz[i], Fy2[i], Fz2[i])
        susp_np = susp.compute_forces(state_np)
        gf_np = assembler.assemble_GF_SYSTEM(state_np, susp_np, wr_np, fastener_forces=None, subrail_forces=None)

        xt = torch.as_tensor(xcar, dtype=torch.float32).view(1, 1, 35)
        vt = torch.as_tensor(vcar, dtype=torch.float32).view(1, 1, 35)
        st = tfm._extract_vehicle_state_torch(xt, vt)
        susp_t = tfm._compute_passenger_suspension_forces_torch(st, dyn_ctx)
        Fy_t = torch.as_tensor(Fy[i], dtype=torch.float32).view(1, 1, 8)
        Fz_t = torch.as_tensor(Fz[i], dtype=torch.float32).view(1, 1, 8)
        Fy2_t = torch.as_tensor(Fy2[i], dtype=torch.float32).view(1, 1, 8)
        Fz2_t = torch.as_tensor(Fz2[i], dtype=torch.float32).view(1, 1, 8)
        gf_t = tfm._assemble_vehicle_gf_torch(st, susp_t, Fz_t, Fy_t, Fz2_t, Fy2_t, dyn_ctx).detach().cpu().numpy().reshape(-1)

        d_gf = gf_t - gf_np
        gf_absmax.append(np.max(np.abs(d_gf)))
        gf_rmse.append(np.sqrt(np.mean(d_gf**2)))

        m = dyn_ctx.mass_vehicle.astype(np.float64)
        a_np = gf_np / m
        a_t = gf_t.astype(np.float64) / m
        d_a = a_t - a_np
        a_absmax.append(np.max(np.abs(d_a)))
        a_rmse.append(np.sqrt(np.mean(d_a**2)))

        j = np.where(idxs == i)[0][0]
        a_np_stack[j, :] = a_np
        a_t_stack[j, :] = a_t
        a_saved_stack[j, :] = A[i, :35]

        d_as = a_t - A[i, :35]
        a_saved_absmax.append(np.max(np.abs(d_as)))
        a_saved_rmse.append(np.sqrt(np.mean(d_as**2)))

    gf_absmax = np.asarray(gf_absmax)
    gf_rmse = np.asarray(gf_rmse)
    a_absmax = np.asarray(a_absmax)
    a_rmse = np.asarray(a_rmse)
    a_saved_absmax = np.asarray(a_saved_absmax)
    a_saved_rmse = np.asarray(a_saved_rmse)

    print("=== 对比结果（PINO移植链路 vs generate_main车辆链路）===")
    print(f"样本点数: {len(idxs)}")
    print(f"GF 最大绝对误差: mean={gf_absmax.mean():.6e}, p95={np.percentile(gf_absmax,95):.6e}, max={gf_absmax.max():.6e}")
    print(f"GF RMSE:         mean={gf_rmse.mean():.6e}, p95={np.percentile(gf_rmse,95):.6e}, max={gf_rmse.max():.6e}")
    print(f"A  最大绝对误差: mean={a_absmax.mean():.6e}, p95={np.percentile(a_absmax,95):.6e}, max={a_absmax.max():.6e}")
    print(f"A  RMSE:         mean={a_rmse.mean():.6e}, p95={np.percentile(a_rmse,95):.6e}, max={a_rmse.max():.6e}")

    print("=== 与存档A(完整求解器输出)差异（诊断未建模项）===")
    print(f"A_saved 最大绝对误差: mean={a_saved_absmax.mean():.6e}, p95={np.percentile(a_saved_absmax,95):.6e}, max={a_saved_absmax.max():.6e}")
    print(f"A_saved RMSE:         mean={a_saved_rmse.mean():.6e}, p95={np.percentile(a_saved_rmse,95):.6e}, max={a_saved_rmse.max():.6e}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 图1：逐样本误差
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 4.5), dpi=140)
    x = np.arange(len(idxs))
    ax1.plot(x, a_absmax, label="A max|err| (PINO vs generate_main)", lw=1.6)
    ax1.plot(x, a_rmse, label="A RMSE (PINO vs generate_main)", lw=1.6)
    ax1.plot(x, a_saved_rmse, label="A RMSE (PINO vs saved A)", lw=1.4, alpha=0.85)
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Error")
    ax1.set_title("Dynamics consistency error across sampled steps")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1_path = save_dir / "consistency_error_curves.png"
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # 图2：关键DOF加速度对比（车体/构架/轮对垂向）
    dof_names = [
        (1, "Carbody Z"),
        (6, "Bogie1 Z"),
        (11, "Bogie2 Z"),
        (16, "Wheelset1 Z"),
        (21, "Wheelset2 Z"),
        (26, "Wheelset3 Z"),
        (31, "Wheelset4 Z"),
    ]
    fig2, axes = plt.subplots(len(dof_names), 1, figsize=(11, 2.2 * len(dof_names)), dpi=140, sharex=True)
    if len(dof_names) == 1:
        axes = [axes]
    for ax, (dof, name) in zip(axes, dof_names):
        y_main = a_np_stack[:, dof]
        y_pino = a_t_stack[:, dof]
        y_saved = a_saved_stack[:, dof]

        # 先画 saved A（浅灰虚线）作为背景，再画两条主链路曲线，避免“被遮挡看不见”。
        ax.plot(y_saved, label="saved A", lw=1.4, color="#7f7f7f", alpha=0.55, linestyle="--", zorder=1)
        ax.plot(y_main, label="generate_main chain", lw=1.8, color="#1f77b4", zorder=3)
        ax.plot(y_pino, label="PINO embedded chain", lw=1.4, color="#ff7f0e", linestyle=":", zorder=4)

        max_main_pino = float(np.max(np.abs(y_main - y_pino)))
        max_main_saved = float(np.max(np.abs(y_main - y_saved)))
        ax.text(
            0.01,
            0.92,
            f"max|Δ(main-pino)|={max_main_pino:.2e}  max|Δ(main-saved)|={max_main_saved:.2e}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Sample index")
    fig2.suptitle("Acceleration comparison on key DOFs", y=0.995)
    fig2.tight_layout()
    fig2_path = save_dir / "consistency_key_dof_acc.png"
    fig2.savefig(fig2_path)
    plt.close(fig2)

    print(f"[FIG] saved: {fig1_path}")
    print(f"[FIG] saved: {fig2_path}")


if __name__ == "__main__":
    main()
