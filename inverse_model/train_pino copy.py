'''
Author: 2099193635 2099193635@qq.com
Date: 2026-04-23 06:37:27
LastEditors: 2099193635 2099193635@qq.com
LastEditTime: 2026-04-23 14:51:32
FilePath: /VTCM_PYTHON/inverse_model/train_pino.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
"""
PINO / FNO 训练脚本
从 inverse_dataset_gen 导出的 HDF5 数据集训练 FNO 模型。
输入：车辆传感器加速度 y [T, n_sensors]
输出：轨道不平顺激励 u [T, n_irr_dofs]
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ── 路径配置 ──────────────────────────────────────────────────────────────────
_DIR  = Path(__file__).parent
_ROOT = _DIR.parent

# physicsnemo 优先使用工作区本地版本
_PHYSICSNEMO = str(_ROOT / "physicsnemo")
if _PHYSICSNEMO not in sys.path:
    sys.path.insert(0, _PHYSICSNEMO)
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from physicsnemo.models.fno import FNO
from physicsnemo.utils.checkpoint import save_checkpoint


# ── 数据集 ────────────────────────────────────────────────────────────────────
class InverseH5Dataset(Dataset):
    """
    从 inverse_dataset_gen 导出的 HDF5 读取样本。
    输入 y 已归一化；输出 phys_z0 / phys_x 在 __init__ 中按全局 mean/std 归一化。
    out_norm_stats: 可传入训练集统计量 dict(mean, std) 供验证集复用；
                    传 None 时自动从当前数据计算（应仅在训练集上调用）。
    """

    def __init__(self, h5_path: str | Path, out_norm_stats: dict | None = None):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            self.y        = np.asarray(f["y"],       dtype=np.float32)  # [N, T, 1]
            self.u        = np.asarray(f["u"],       dtype=np.float32)  # [N, T, 1]
            self.c        = np.asarray(f["c"],       dtype=np.float32)  # [N, n_cond]
            self.x_query  = np.asarray(f["x_query"], dtype=np.float32)  # [N, T]
            phys_z0_raw   = np.asarray(f["phys_z0"], dtype=np.float32)  # [N, T, 1] m
            phys_x_raw    = np.asarray(f["phys_x"],  dtype=np.float32)  # [N, T, 7] m
            self.phys_v   = np.asarray(f["phys_v"],  dtype=np.float32)  # [N, T, 7] m/s
            self.phys_a   = np.asarray(f["phys_a"],  dtype=np.float32)  # [N, T, 7] m/s²
            self.vx       = np.asarray(f["vx"],      dtype=np.float32)  # [N, 1] km/h
            self.seq_lengths = (
                np.asarray(f["seq_lengths"], dtype=np.int64)
                if "seq_lengths" in f else None
            )

        # ── 拼合输出为 [N, T, 8]：[phys_z0 | phys_x] ──────────────────────
        out_raw = np.concatenate([phys_z0_raw, phys_x_raw], axis=-1)  # [N, T, 8]

        # ── 归一化统计量（8 通道）─────────────────────────────────────────
        if out_norm_stats is None:
            flat = out_raw.reshape(-1, 8)          # [N*T, 8]
            mean = flat.mean(axis=0).astype(np.float32)  # (8,)
            std  = flat.std(axis=0).astype(np.float32)
            std  = np.where(std < 1e-12, 1.0, std)
            self.out_norm_stats = {"mean": mean, "std": std}
        else:
            self.out_norm_stats = out_norm_stats

        m = self.out_norm_stats["mean"]   # (8,)
        s = self.out_norm_stats["std"]    # (8,)
        self.out_norm = ((out_raw - m) / s).astype(np.float32)  # [N, T, 8] 归一化输出

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "y":       torch.from_numpy(self.y[idx]),
            "u":       torch.from_numpy(self.u[idx]),
            "c":       torch.from_numpy(self.c[idx]),
            "x_query": torch.from_numpy(self.x_query[idx]),
            "out_norm": torch.from_numpy(self.out_norm[idx]),   # [T, 8] 归一化输出目标
            "phys_v":  torch.from_numpy(self.phys_v[idx]),
            "phys_a":  torch.from_numpy(self.phys_a[idx]),
            "vx":      torch.from_numpy(self.vx[idx]),
        }
        if self.seq_lengths is not None:
            sample["seq_lengths"] = torch.tensor(self.seq_lengths[idx], dtype=torch.long)
        return sample


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PINO/FNO Training for VTCM Inverse Problem")
    p.add_argument("--epochs",             type=int,   default=200)
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--lr",                 type=float, default=3e-4)
    p.add_argument("--gamma",              type=float, default=0.99948708,
                   help="ExponentialLR 衰减系数")
    p.add_argument("--latent_channels",    type=int,   default=64)
    p.add_argument("--num_fno_layers",     type=int,   default=6)
    p.add_argument("--num_fno_modes",      type=int,   default=12)
    p.add_argument("--decoder_layers",     type=int,   default=1)
    p.add_argument("--decoder_layer_size", type=int,   default=64)
    p.add_argument("--padding",            type=int,   default=9)
    p.add_argument("--ckpt_dir",           type=str,   default=str(_DIR / "checkpoints_pino"))
    p.add_argument("--fig_dir",            type=str,   default=str(_DIR / "figures_pino"))
    p.add_argument("--vis_every",          type=int,   default=10,
                   help="每 vis_every 个 epoch 保存验证可视化图")
    p.add_argument("--dataset_dir",        type=str,
                   default=str(_ROOT / "datasets" / "VTCM_inverse"),
                   help="inverse_dataset_gen 导出的数据集目录")
    p.add_argument("--train-full-seq", dest="train_full_seq",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="训练是否使用 *_full_seq.hdf5")
    p.add_argument("--val-full-seq",   dest="val_full_seq",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="验证是否使用 *_full_seq.hdf5")
    return p.parse_args(argv)
    

# ── 通道标签（与模型输出顺序一致） ───────────────────────────────────────────────
# 模型预测 8 通道：[0] phys_z0(m)，[1-7] phys_x Zc/Zt1/Zt2/Zw1/Zw2/Zw3/Zw4(m)
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


def _build_target(batch: dict, device: torch.device) -> torch.Tensor:
    """从 batch 取归一化后的 8 通道输出目标 [B, 8, T]。"""
    return batch["out_norm"].to(device, non_blocking=True).permute(0, 2, 1)  # [B, 8, T]


# ── 绘图公共设置 ──────────────────────────────────────────────────────────────
def _plot_rcparams():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 11, "axes.labelsize": 11,
        "axes.titlesize": 11, "legend.fontsize": 9,
        "lines.linewidth": 1.4,
    })


# ── 验证步骤 & 可视化 ─────────────────────────────────────────────────────────
def test_step(model: torch.nn.Module, dataloader: DataLoader,
              epoch: int, device: torch.device,
              fig_dir: str, vis_every: int,
              out_norm_stats: dict) -> float:
    model.eval()
    loss_sum = 0.0
    n_batch  = 0
    viz_pred_norm   = None
    viz_target_norm = None

    with torch.no_grad():
        for batch in dataloader:
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)  # [B, 1, T]
            target = _build_target(batch, device)                               # [B, 8, T]
            pred   = model(inp)                                                  # [B, 8, T]
            loss_sum += F.mse_loss(pred, target).item()
            n_batch  += 1
            if viz_pred_norm is None:
                viz_pred_norm   = pred[0].detach().cpu().numpy()    # [8, T]
                viz_target_norm = target[0].detach().cpu().numpy()  # [8, T]

    avg_loss = loss_sum / max(n_batch, 1)

    if viz_pred_norm is not None and (epoch % vis_every == 0):
        os.makedirs(fig_dir, exist_ok=True)
        _plot_rcparams()
        # 反归一化到物理量纲
        m = out_norm_stats["mean"][:, np.newaxis]  # (8, 1)
        s = out_norm_stats["std"][:, np.newaxis]   # (8, 1)
        viz_pred   = viz_pred_norm   * s + m       # [8, T]
        viz_target = viz_target_norm * s + m       # [8, T]

        T      = viz_pred.shape[1]
        dt_s   = 1e-4
        t_axis = np.arange(T) * dt_s

        # 分两组：Z0 (1 行) + 7-DOF (7 行)
        groups = [
            {"title": "Track Irregularity (Z0)",
             "ch_ids": [0], "ylabel": "Displacement (m)"},
            {"title": "Vehicle State - 7 DOF Displacements",
             "ch_ids": list(range(1, 8)), "ylabel": "Displacement (m)"},
        ]
        for grp in groups:
            ch_ids = grp["ch_ids"]
            n_ch   = len(ch_ids)
            fig, axes = plt.subplots(n_ch, 1, figsize=(10, 3.0 * n_ch),
                                     constrained_layout=True,
                                     squeeze=False)
            for row, ch_i in enumerate(ch_ids):
                ax  = axes[row, 0]
                sig_pred   = viz_pred[ch_i]
                sig_target = viz_target[ch_i]
                rmse = float(np.sqrt(np.mean((sig_pred - sig_target) ** 2)))

                ax.plot(t_axis, sig_target, color="#1f77b4", lw=1.2,
                        linestyle="--", label="Ground Truth")
                ax.plot(t_axis, sig_pred,   color="#d62728", lw=1.2,
                        label=f"Prediction  RMSE={rmse:.3e}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(grp["ylabel"])
                ax.set_title(OUT_CH_LABELS[ch_i])
                ax.legend(frameon=False, ncol=2)

            safe = grp["title"].replace(" ", "_").replace("/", "")
            fig.suptitle(f"{grp['title']}  |  Val Epoch {epoch + 1}",
                         fontsize=12, fontweight="bold")
            fig.savefig(
                os.path.join(fig_dir, f"val_epoch_{epoch:04d}_{safe}.png"),
                dpi=200, bbox_inches="tight",
            )
            plt.close(fig)

    return avg_loss


# ── 训练结束后全量推理可视化 ───────────────────────────────────────────────────
def inference_and_plot(model: torch.nn.Module, dataloader: DataLoader,
                       device: torch.device, fig_dir: str, epoch: int,
                       out_norm_stats: dict, max_samples: int = 4) -> None:
    """推理并绘制 8 通道输出的时序对比图（物理量纲）。"""
    model.eval()
    os.makedirs(fig_dir, exist_ok=True)
    _plot_rcparams()
    m   = out_norm_stats["mean"][:, np.newaxis]  # (8, 1)
    s   = out_norm_stats["std"][:, np.newaxis]   # (8, 1)
    dt_s = 1e-4
    sample_num = 0
    with torch.no_grad():
        for batch in dataloader:
            if sample_num >= max_samples:
                break
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)  # [B, 1, T]
            target = _build_target(batch, device)                               # [B, 8, T]
            pred   = model(inp)                                                  # [B, 8, T]

            for i in range(min(pred.shape[0], max_samples - sample_num)):
                T      = pred.shape[2]
                t_axis = np.arange(T) * dt_s
                # 反归一化
                p_phys = pred[i].cpu().numpy()   * s + m    # [8, T]
                t_phys = target[i].cpu().numpy() * s + m    # [8, T]

                fig, axes = plt.subplots(8, 1, figsize=(10, 3.0 * 8),
                                         constrained_layout=True, squeeze=False)
                for ch_i in range(8):
                    ax   = axes[ch_i, 0]
                    rmse = float(np.sqrt(np.mean((p_phys[ch_i] - t_phys[ch_i]) ** 2)))
                    ax.plot(t_axis, t_phys[ch_i], color="#1f77b4", lw=1.2,
                            linestyle="--", label="Ground Truth")
                    ax.plot(t_axis, p_phys[ch_i], color="#d62728", lw=1.2,
                            label=f"Prediction  RMSE={rmse:.3e}")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Displacement (m)")
                    ax.set_title(OUT_CH_LABELS[ch_i])
                    ax.legend(frameon=False, ncol=2)

                fig.suptitle(
                    f"Inference @ Epoch {epoch + 1}  |  Sample {sample_num}",
                    fontsize=13, fontweight="bold",
                )
                fig.savefig(
                    os.path.join(fig_dir, f"infer_epoch_{epoch:04d}_sample_{sample_num:04d}.png"),
                    dpi=200, bbox_inches="tight",
                )
                plt.close(fig)
                sample_num += 1


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir  = Path(args.dataset_dir)
    train_suffix = "_full_seq" if args.train_full_seq else ""
    val_suffix   = "_full_seq" if args.val_full_seq   else ""
    train_h5 = dataset_dir / f"train{train_suffix}.hdf5"
    val_h5   = dataset_dir / f"validation{val_suffix}.hdf5"

    # 训练集：自动计算归一化统计；验证集复用训练集统计
    train_ds = InverseH5Dataset(train_h5, out_norm_stats=None)
    val_ds   = InverseH5Dataset(val_h5,   out_norm_stats=train_ds.out_norm_stats)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True,
    ) if len(val_ds) > 0 else None

    out_norm_stats = train_ds.out_norm_stats
    # 输入：车体加速度 y (1ch)；输出：phys_z0 + phys_x 7-DOF = 8ch（均已归一化）
    in_channels  = train_ds[0]["y"].shape[-1]        # 1
    out_channels = train_ds[0]["out_norm"].shape[-1]  # 8

    print("=" * 80)
    print(f"Device       : {device}")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"in_channels={in_channels}, out_channels={out_channels}")
    print(f"  → 输出通道: [0] phys_z0, [1-7] phys_x (Zc/Zt1/Zt2/Zw1-Zw4)")
    print(f"输出归一化 mean: {out_norm_stats['mean']}")
    print(f"输出归一化 std : {out_norm_stats['std']}")
    print(f"Train HDF5   : {train_h5}")
    print(f"Val   HDF5   : {val_h5}")
    print("=" * 80)

    model = FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        decoder_layers=args.decoder_layers,
        decoder_layer_size=args.decoder_layer_size,
        dimension=1,
        latent_channels=args.latent_channels,
        num_fno_layers=args.num_fno_layers,
        num_fno_modes=args.num_fno_modes,
        padding=args.padding,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for batch in pbar:
            optimizer.zero_grad()
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)  # [B, 1, T]
            target = _build_target(batch, device)                               # [B, 8, T]
            pred   = model(inp)                                                  # [B, 8, T]
            loss   = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4e}")

        avg_loss = total_loss / len(train_loader)
        val_loss = 0.0
        if val_loader is not None:
            val_loss = test_step(
                model, val_loader, epoch, device,
                args.fig_dir, args.vis_every, out_norm_stats,
            )

        print(
            f"[Epoch {epoch + 1:4d}/{args.epochs}] "
            f"train={avg_loss:.4e}  val={val_loss:.4e}  "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        save_checkpoint(
            args.ckpt_dir, models=model,
            optimizer=optimizer, scheduler=scheduler, epoch=epoch,
        )

    # 训练结束后全量推理可视化
    if val_loader is not None:
        print("Generating full inference plots ...")
        inference_and_plot(model, val_loader, device, args.fig_dir,
                           epoch=args.epochs - 1,
                           out_norm_stats=out_norm_stats, max_samples=8)
        print(f"Figures saved to: {args.fig_dir}")


if __name__ == "__main__":
    main()
