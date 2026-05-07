"""inverse_model/utils/viz.py
绘图工具函数，供各模型训练脚本共用。
通道布局（8 通道）：[0]=Z0, [1]=Zc, [2]=Zt1, [3]=Zt2, [4-7]=Zw1-Zw4
"""
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_CH_LABELS = [
    "Z₀ (wheel-1 irreg, m)",
    "Zc  (car body, m)",
    "Zt1 (bogie front, m)",
    "Zt2 (bogie rear, m)",
    "Zw1 (wheelset 1, m)",
    "Zw2 (wheelset 2, m)",
    "Zw3 (wheelset 3, m)",
    "Zw4 (wheelset 4, m)",
]

_VIZ_GROUPS = [
    {"title": "Track Irregularity (Z0)",          "ch_ids": [0],          "ylabel": "Displacement (m)"},
    {"title": "Vehicle State - 7 DOF Displacements", "ch_ids": list(range(1, 8)), "ylabel": "Displacement (m)"},
]


def _setup_rcparams():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 11,
        "legend.fontsize": 9, "lines.linewidth": 1.4,
    })


def _denorm_np(arr_norm: np.ndarray, out_norm_stats: dict) -> np.ndarray:
    """arr_norm: [C, T] → physical"""
    m = out_norm_stats["mean"][:, np.newaxis]
    s = out_norm_stats["std"][:, np.newaxis]
    return arr_norm * s + m


def plot_val_predictions(pred_norm: np.ndarray, target_norm: np.ndarray,
                         out_norm_stats: dict, fig_dir: str,
                         epoch: int, dt: float = 1e-4):
    """
    保存验证集首个样本的时序对比图（按通道分组）。
    pred_norm / target_norm: [C, T] numpy 数组（归一化空间）。
    """
    os.makedirs(fig_dir, exist_ok=True)
    _setup_rcparams()
    pred   = _denorm_np(pred_norm,   out_norm_stats)
    target = _denorm_np(target_norm, out_norm_stats)
    T      = pred.shape[1]
    t_axis = np.arange(T) * dt

    for grp in _VIZ_GROUPS:
        ch_ids = grp["ch_ids"]
        fig, axes = plt.subplots(len(ch_ids), 1,
                                 figsize=(10, 3.0 * len(ch_ids)),
                                 constrained_layout=True, squeeze=False)
        for row, ch_i in enumerate(ch_ids):
            ax   = axes[row, 0]
            rmse = float(np.sqrt(np.mean((pred[ch_i] - target[ch_i]) ** 2)))
            ax.plot(t_axis, target[ch_i], color="#1f77b4", lw=1.2, ls="--", label="Ground Truth")
            ax.plot(t_axis, pred[ch_i],   color="#d62728", lw=1.2, label=f"Pred  RMSE={rmse:.3e}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel(grp["ylabel"])
            ax.set_title(OUT_CH_LABELS[ch_i]); ax.legend(frameon=False, ncol=2)
        safe = grp["title"].replace(" ", "_").replace("/", "")
        fig.suptitle(f"{grp['title']}  |  Val Epoch {epoch + 1}", fontsize=12, fontweight="bold")
        fig.savefig(os.path.join(fig_dir, f"val_epoch_{epoch:04d}_{safe}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_inference_samples(pred_list: list, target_list: list,
                           out_norm_stats: dict, fig_dir: str,
                           epoch: int, dt: float = 1e-4):
    """
    保存多个推理样本的全通道时序对比图。
    pred_list / target_list: list of [C, T] numpy 数组。
    """
    os.makedirs(fig_dir, exist_ok=True)
    _setup_rcparams()

    for si, (pn, tn) in enumerate(zip(pred_list, target_list)):
        p = _denorm_np(pn, out_norm_stats)
        t = _denorm_np(tn, out_norm_stats)
        T = p.shape[1]
        t_axis = np.arange(T) * dt

        fig, axes = plt.subplots(8, 1, figsize=(10, 24),
                                 constrained_layout=True, squeeze=False)
        for ch_i in range(8):
            ax   = axes[ch_i, 0]
            rmse = float(np.sqrt(np.mean((p[ch_i] - t[ch_i]) ** 2)))
            ax.plot(t_axis, t[ch_i], color="#1f77b4", lw=1.2, ls="--", label="Ground Truth")
            ax.plot(t_axis, p[ch_i], color="#d62728", lw=1.2, label=f"Pred  RMSE={rmse:.3e}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Displacement (m)")
            ax.set_title(OUT_CH_LABELS[ch_i]); ax.legend(frameon=False, ncol=2)
        fig.suptitle(f"Inference @ Epoch {epoch + 1}  |  Sample {si}", fontsize=13, fontweight="bold")
        fig.savefig(os.path.join(fig_dir, f"infer_epoch_{epoch:04d}_sample_{si:04d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
