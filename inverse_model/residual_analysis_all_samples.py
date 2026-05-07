"""
Residual analysis for all samples in datasets/VTCM_inverse/train_full_seq.hdf5

Outputs:
1) CSV metrics per sample
2) High-quality SCI-style figure (English labels, Times New Roman)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Ensure project root import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.parameters import VehicleParams  # noqa: E402


def time_shift(arr: np.ndarray, steps: int) -> np.ndarray:
    """Shift time series with boundary padding.
    steps > 0: shift right (use first value for left padding)
    steps < 0: shift left  (use last value for right padding)
    """
    if steps == 0:
        return arr.copy()
    out = np.empty_like(arr)
    if steps > 0:
        out[:steps] = arr[0]
        out[steps:] = arr[:-steps]
    else:
        s = -steps
        out[-s:] = arr[-1]
        out[:-s] = arr[s:]
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def main() -> None:
    # ----------------------------- paths -----------------------------
    h5_path = ROOT / "datasets" / "VTCM_inverse" / "train_full_seq.hdf5"
    norm_path = ROOT / "datasets" / "VTCM_inverse" / "norm_stats.npz"
    out_dir = ROOT / "results" / "residual_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------- plotting style ----------------------
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "savefig.dpi": 600,
            "figure.dpi": 150,
        }
    )

    # ----------------------- physical parameters ---------------------
    veh = VehicleParams(vehicle_type="高速客车", yaml_dir="configs/standard")
    Mc, Mt, Mw = float(veh.Mc), float(veh.Mt), float(veh.Mw)
    Ksz, Csz = float(veh.Ksz), float(veh.Csz)
    Kpz, Cpz = float(veh.Kpz), float(veh.Cpz)
    Lc, Lt = float(veh.Lc), float(veh.Lt)
    G_hertz = 4.5e-8

    # vehicle-mode vertical DOF indices
    idx = {"Zc": 1, "Zt1": 6, "Zt2": 11, "Zw1": 16, "Zw2": 21, "Zw3": 26, "Zw4": 31}

    # -------------------------- load dataset -------------------------
    ns = np.load(norm_path)
    y_mean = float(ns["y_mean"].flat[0])
    y_std = float(ns["y_std"].flat[0])
    u_mean = float(ns["u_mean"].flat[0])
    u_std = float(ns["u_std"].flat[0])

    with h5py.File(h5_path, "r") as f:
        y_norm = f["y"][:]  # [N, T, 1]
        u_norm = f["u"][:]  # [N, T, 1]
        vx_all = f["vx"][:]  # [N, 1]
        seq_lengths = f["seq_lengths"][:]  # [N]
        source_file = f["source_file"][:]  # [N]
        skip_initial_seconds = float(f.attrs.get("skip_initial_seconds", 2.0))

    n_samples = y_norm.shape[0]

    metric_names = [
        "car_body_constraint_npz",
        "car_body_constraint_hdf5",
        "bogie_front",
        "bogie_rear",
        "nexus_w1",
        "nexus_w2",
        "nexus_w3",
        "nexus_w4",
    ]

    metrics_rms: Dict[str, List[float]] = {k: [] for k in metric_names}
    metrics_mean: Dict[str, List[float]] = {k: [] for k in metric_names}
    metrics_maxabs: Dict[str, List[float]] = {k: [] for k in metric_names}

    rep_time = None
    rep_res_npz = None
    rep_bogie_front = None
    rep_nexus = None

    for i in range(n_samples):
        src = source_file[i]
        if isinstance(src, (bytes, np.bytes_)):
            src = src.decode("utf-8", errors="ignore")
        src = str(src)

        npz_path = Path(src)
        if not npz_path.is_absolute():
            npz_path = ROOT / npz_path
        if not npz_path.exists():
            # record NaN if source file missing
            for k in metric_names:
                metrics_rms[k].append(np.nan)
                metrics_mean[k].append(np.nan)
                metrics_maxabs[k].append(np.nan)
            continue

        data = np.load(npz_path, allow_pickle=True)
        dt = float(data["dt"])
        lead_steps = int(round(skip_initial_seconds / dt))

        seq_len = int(seq_lengths[i])
        sl = slice(lead_steps, lead_steps + seq_len)

        X = data["X"]
        V = data["V"]
        A = data["A"]
        if sl.stop > X.shape[0]:
            seq_len = max(0, X.shape[0] - lead_steps)
            sl = slice(lead_steps, lead_steps + seq_len)
            if seq_len <= 0:
                for k in metric_names:
                    metrics_rms[k].append(np.nan)
                    metrics_mean[k].append(np.nan)
                    metrics_maxabs[k].append(np.nan)
                continue

        # states from NPZ (float64)
        Zc_x, Zc_v, Zc_a = X[sl, idx["Zc"]], V[sl, idx["Zc"]], A[sl, idx["Zc"]]
        Zt1_x, Zt1_v, Zt1_a = X[sl, idx["Zt1"]], V[sl, idx["Zt1"]], A[sl, idx["Zt1"]]
        Zt2_x, Zt2_v, Zt2_a = X[sl, idx["Zt2"]], V[sl, idx["Zt2"]], A[sl, idx["Zt2"]]
        Zw1_x, Zw1_v, Zw1_a = X[sl, idx["Zw1"]], V[sl, idx["Zw1"]], A[sl, idx["Zw1"]]
        Zw2_x, Zw2_v, Zw2_a = X[sl, idx["Zw2"]], V[sl, idx["Zw2"]], A[sl, idx["Zw2"]]
        Zw3_x, Zw3_v, Zw3_a = X[sl, idx["Zw3"]], V[sl, idx["Zw3"]], A[sl, idx["Zw3"]]
        Zw4_x, Zw4_v, Zw4_a = X[sl, idx["Zw4"]], V[sl, idx["Zw4"]], A[sl, idx["Zw4"]]

        # de-normalized measurement signals
        acc_c = y_norm[i, :seq_len, 0].astype(np.float64) * y_std + y_mean
        Z0 = u_norm[i, :seq_len, 0].astype(np.float64) * u_std + u_mean

        vx = float(vx_all[i, 0]) if np.ndim(vx_all[i]) else float(vx_all[i])
        # phys_z0/u = bz_L[3,:] = 第4轮对（最后位）基准。
        # 轮对1/2/3 在第4轮对之前，取未来值（负步数 = 向左平移）。
        # 步数 = 2*spacing/V/dt（与 irregularity.py stack_wheelsets 一致）
        dt1_steps = int(round(2 * (Lc + Lt) / vx / dt))  # 轮对4→轮对1
        dt2_steps = int(round(2 * Lc         / vx / dt))  # 轮对4→轮对2
        dt3_steps = int(round(2 * Lt         / vx / dt))  # 轮对4→轮对3

        Z0_w1 = time_shift(Z0, -dt1_steps)   # 向左取未来
        Z0_w2 = time_shift(Z0, -dt2_steps)
        Z0_w3 = time_shift(Z0, -dt3_steps)
        Z0_w4 = Z0                            # 轮对4即基准，无需平移

        F_sz1 = Ksz * (Zc_x - Zt1_x) + Csz * (Zc_v - Zt1_v)
        F_sz2 = Ksz * (Zc_x - Zt2_x) + Csz * (Zc_v - Zt2_v)
        F_pz1 = Kpz * (Zt1_x - Zw1_x) + Cpz * (Zt1_v - Zw1_v)
        F_pz2 = Kpz * (Zt1_x - Zw2_x) + Cpz * (Zt1_v - Zw2_v)
        F_pz3 = Kpz * (Zt2_x - Zw3_x) + Cpz * (Zt2_v - Zw3_v)
        F_pz4 = Kpz * (Zt2_x - Zw4_x) + Cpz * (Zt2_v - Zw4_v)

        P1 = F_pz1 - 0.5 * Mw * Zw1_a
        P2 = F_pz2 - 0.5 * Mw * Zw2_a
        P3 = F_pz3 - 0.5 * Mw * Zw3_a
        P4 = F_pz4 - 0.5 * Mw * Zw4_a

        dz_kin1 = G_hertz * np.maximum(P1, 0.0) ** (2.0 / 3.0)
        dz_kin2 = G_hertz * np.maximum(P2, 0.0) ** (2.0 / 3.0)
        dz_kin3 = G_hertz * np.maximum(P3, 0.0) ** (2.0 / 3.0)
        dz_kin4 = G_hertz * np.maximum(P4, 0.0) ** (2.0 / 3.0)

        r = {
            "car_body_constraint_npz": Mc * Zc_a + 2.0 * F_sz1 + 2.0 * F_sz2,
            "car_body_constraint_hdf5": Mc * acc_c + 2.0 * F_sz1 + 2.0 * F_sz2,
            "bogie_front": Mt * Zt1_a - 2.0 * F_sz1 + 2.0 * F_pz1 + 2.0 * F_pz2,
            "bogie_rear": Mt * Zt2_a - 2.0 * F_sz2 + 2.0 * F_pz3 + 2.0 * F_pz4,
            "nexus_w1": dz_kin1 - (Zw1_x - Z0_w1),
            "nexus_w2": dz_kin2 - (Zw2_x - Z0_w2),
            "nexus_w3": dz_kin3 - (Zw3_x - Z0_w3),
            "nexus_w4": dz_kin4 - (Zw4_x - Z0_w4),
        }

        for k in metric_names:
            rk = r[k]
            metrics_rms[k].append(rms(rk))
            metrics_mean[k].append(float(np.mean(rk)))
            metrics_maxabs[k].append(float(np.max(np.abs(rk))))

        if rep_time is None:
            rep_time = np.arange(seq_len) * dt
            rep_res_npz = r["car_body_constraint_npz"]
            rep_bogie_front = r["bogie_front"]
            rep_nexus = r["nexus_w1"]

    # -------------------------- save CSV -----------------------------
    csv_path = out_dir / "residual_metrics_all_samples.csv"
    header = ["sample_index", "source_file"]
    for k in metric_names:
        header += [f"{k}_mean", f"{k}_rms", f"{k}_maxabs"]

    with open(csv_path, "w", encoding="utf-8") as wf:
        wf.write(",".join(header) + "\n")
        for i in range(n_samples):
            src = source_file[i]
            if isinstance(src, (bytes, np.bytes_)):
                src = src.decode("utf-8", errors="ignore")
            row = [str(i), '"' + str(src).replace('"', "'") + '"']
            for k in metric_names:
                row.append(f"{metrics_mean[k][i]:.12e}" if np.isfinite(metrics_mean[k][i]) else "nan")
                row.append(f"{metrics_rms[k][i]:.12e}" if np.isfinite(metrics_rms[k][i]) else "nan")
                row.append(f"{metrics_maxabs[k][i]:.12e}" if np.isfinite(metrics_maxabs[k][i]) else "nan")
            wf.write(",".join(row) + "\n")

    # -------------------------- summary print ------------------------
    print("=" * 78)
    print(f"Processed samples: {n_samples}")
    print(f"Metrics CSV      : {csv_path}")
    for k in metric_names:
        arr = np.array(metrics_rms[k], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        print(
            f"{k:30s}  median RMS={np.median(arr):.4e}, "
            f"p95 RMS={np.percentile(arr, 95):.4e}, max RMS={np.max(arr):.4e}"
        )
    print("=" * 78)

    # -------------------------- plotting -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)

    # Panel A: boxplot of RMS (force-equations)
    ax = axes[0, 0]
    force_keys = ["car_body_constraint_npz", "bogie_front", "bogie_rear"]
    force_data = [np.array(metrics_rms[k], dtype=float) for k in force_keys]
    force_data = [d[np.isfinite(d)] for d in force_data]
    labels = ["CarBody", "Bogie Front", "Bogie Rear"]
    ax.boxplot(force_data, tick_labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("RMS residual (N)")
    ax.set_title("(a) Force-equation residual RMS")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    # Panel B: boxplot of RMS (contact equations)
    ax = axes[0, 1]
    contact_keys = ["nexus_w1", "nexus_w2", "nexus_w3", "nexus_w4"]
    contact_data = [np.array(metrics_rms[k], dtype=float) for k in contact_keys]
    contact_data = [d[np.isfinite(d)] for d in contact_data]
    ax.boxplot(contact_data, tick_labels=["W1", "W2", "W3", "W4"], showfliers=False)
    ax.set_ylabel("RMS residual (m)")
    ax.set_title("(b) Contact-constraint residual RMS")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    # Panel C: histogram of car-body NPZ residual RMS
    ax = axes[1, 0]
    car_npz = np.array(metrics_rms["car_body_constraint_npz"], dtype=float)
    car_npz = car_npz[np.isfinite(car_npz)]
    ax.hist(car_npz, bins=24, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.set_xlabel("RMS residual (N)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Distribution of CarBody(NPZ) RMS")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    # Panel D: representative time-series (sample 0)
    ax = axes[1, 1]
    if rep_time is not None and rep_res_npz is not None and rep_bogie_front is not None and rep_nexus is not None:
        n_show = min(2000, rep_time.size)
        t = rep_time[:n_show]
        ax.plot(t, rep_res_npz[:n_show], label="CarBody(NPZ), N", color="#55A868")
        ax.plot(t, rep_bogie_front[:n_show], label="BogieFront(NPZ), N", color="#C44E52")
        ax2 = ax.twinx()
        ax2.plot(t, rep_nexus[:n_show], label="Nexus W1, m", color="#8172B2", alpha=0.85)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Residual (N)")
        ax2.set_ylabel("Residual (m)")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_title("(d) Representative residual traces")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    else:
        ax.text(0.5, 0.5, "No representative sample available", ha="center", va="center")

    fig_path_png = out_dir / "residual_analysis_sci.png"
    fig_path_pdf = out_dir / "residual_analysis_sci.pdf"
    fig.savefig(fig_path_png, dpi=600, bbox_inches="tight")
    fig.savefig(fig_path_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure PNG       : {fig_path_png}")
    print(f"Figure PDF       : {fig_path_pdf}")


if __name__ == "__main__":
    main()
