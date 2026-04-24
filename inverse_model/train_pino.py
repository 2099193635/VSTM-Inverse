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
    """从 inverse_dataset_gen 导出的 HDF5 读取样本。"""

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            self.y        = np.asarray(f["y"],       dtype=np.float32)  # [N, T, n_s]
            self.u        = np.asarray(f["u"],       dtype=np.float32)  # [N, L, n_dir]
            self.c        = np.asarray(f["c"],       dtype=np.float32)  # [N, n_cond]
            self.x_query  = np.asarray(f["x_query"], dtype=np.float32)  # [N, L]
            self.seq_lengths = (
                np.asarray(f["seq_lengths"], dtype=np.int64)
                if "seq_lengths" in f else None
            )

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "y":       torch.from_numpy(self.y[idx]),
            "u":       torch.from_numpy(self.u[idx]),
            "c":       torch.from_numpy(self.c[idx]),
            "x_query": torch.from_numpy(self.x_query[idx]),
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
    

# ── 验证步骤 & 可视化 ─────────────────────────────────────────────────────────
def test_step(model: torch.nn.Module, dataloader: DataLoader,
              epoch: int, device: torch.device,
              fig_dir: str, vis_every: int) -> float:
    model.eval()
    loss_epoch = 0.0
    n_batch    = 0
    viz_pred   = None
    viz_target = None

    with torch.no_grad():
        for batch in dataloader:
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)  # [B, C_in,  T]
            target = batch["u"].to(device, non_blocking=True).transpose(1, 2)  # [B, C_out, T]
            pred   = model(inp)
            loss_epoch += F.mse_loss(pred, target).item()
            n_batch    += 1
            if viz_pred is None:
                viz_pred   = pred[0].detach().cpu().numpy()    # [C_out, T]
                viz_target = target[0].detach().cpu().numpy()  # [C_out, T]

    avg_loss = loss_epoch / max(n_batch, 1)

    if viz_pred is not None and (epoch % vis_every == 0):
        os.makedirs(fig_dir, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.family": "serif", "font.size": 11,
            "axes.labelsize": 11,   "axes.titlesize": 12,
            "legend.fontsize": 9,   "lines.linewidth": 1.4,
            "figure.dpi": 150,
        })
        n_ch = viz_pred.shape[0]
        T    = viz_pred.shape[1]
        t_axis = np.arange(T)

        # 每个输出通道单独存一张图（左：时序对比，右：PSD 对比）
        for ch_i in range(n_ch):
            sig_pred   = viz_pred[ch_i]    # [T]
            sig_target = viz_target[ch_i]  # [T]

            # ── PSD via Welch ────────────────────────────────────────────────
            from scipy.signal import welch
            nperseg = min(256, T // 4)
            f_pred,   psd_pred   = welch(sig_pred,   nperseg=nperseg)
            f_target, psd_target = welch(sig_target, nperseg=nperseg)

            fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(14, 4))

            # 左子图：时序曲线对比
            ax_t.plot(t_axis, sig_target, color="#1f77b4", linestyle="--",
                      label="Ground Truth", linewidth=1.2)
            ax_t.plot(t_axis, sig_pred,   color="#d62728",
                      label="Prediction",  linewidth=1.2)
            ax_t.set_xlabel("Time Step")
            ax_t.set_ylabel("Amplitude")
            ax_t.set_title(f"Time Series (ch {ch_i})")
            ax_t.legend(frameon=False, loc="upper right")
            ax_t.grid(True, alpha=0.35)
            ax_t.spines["top"].set_visible(False)
            ax_t.spines["right"].set_visible(False)

            # 右子图：功率谱密度对比（半对数纵轴）
            ax_f.semilogy(f_target, psd_target, color="#1f77b4", linestyle="--",
                          label="Ground Truth", linewidth=1.2)
            ax_f.semilogy(f_pred,   psd_pred,   color="#d62728",
                          label="Prediction",   linewidth=1.2)
            ax_f.set_xlabel("Normalised Frequency")
            ax_f.set_ylabel("PSD")
            ax_f.set_title(f"Power Spectral Density (ch {ch_i})")
            ax_f.legend(frameon=False, loc="upper right")
            ax_f.grid(True, alpha=0.35, which="both")
            ax_f.spines["top"].set_visible(False)
            ax_f.spines["right"].set_visible(False)

            fig.suptitle(f"Validation @ Epoch {epoch + 1}  |  Channel {ch_i}", y=1.01)
            fig.tight_layout()
            fig.savefig(
                os.path.join(fig_dir, f"val_epoch_{epoch:04d}_ch{ch_i}.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig)

    return avg_loss


# ── 训练结束后全量推理可视化 ───────────────────────────────────────────────────
def inference_and_plot(model: torch.nn.Module, dataloader: DataLoader,
                       device: torch.device, fig_dir: str, epoch: int) -> None:
    from scipy.signal import welch
    model.eval()
    os.makedirs(fig_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 11,   "axes.titlesize": 12,
        "legend.fontsize": 9,   "lines.linewidth": 1.4,
        "figure.dpi": 150,
    })
    sample_num = 0
    with torch.no_grad():
        for batch in dataloader:
            inp     = batch["y"].to(device, non_blocking=True).transpose(1, 2)
            x_query = batch["x_query"].to(device, non_blocking=True)
            output  = batch["u"].to(device, non_blocking=True).permute(0, 2, 1)  # [B, n_dir, T]
            pred    = model(inp)                                          # [B, n_dir, T]

            for i in range(pred.shape[0]):
                n_ch   = pred.shape[1]
                T      = pred.shape[2]
                t_axis = np.arange(T)

                for ch_i in range(n_ch):
                    sig_pred   = pred[i, ch_i].cpu().numpy()
                    sig_target = output[i, ch_i].cpu().numpy()

                    # PSD via Welch
                    nperseg = min(256, T // 4)
                    f_pred,   psd_pred   = welch(sig_pred,   nperseg=nperseg)
                    f_target, psd_target = welch(sig_target, nperseg=nperseg)

                    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(14, 4))

                    # 左子图：时序曲线对比
                    ax_t.plot(t_axis, sig_target, color="#1f77b4", linestyle="--",
                              label="Ground Truth", linewidth=1.2)
                    ax_t.plot(t_axis, sig_pred,   color="#d62728",
                              label="Prediction",  linewidth=1.2)
                    ax_t.set_xlabel("Time Step")
                    ax_t.set_ylabel("Amplitude")
                    ax_t.set_title(f"Time Series (ch {ch_i})")
                    ax_t.legend(frameon=False, loc="upper right")
                    ax_t.grid(True, alpha=0.35)
                    ax_t.spines["top"].set_visible(False)
                    ax_t.spines["right"].set_visible(False)

                    # 右子图：PSD 对比
                    ax_f.semilogy(f_target, psd_target, color="#1f77b4", linestyle="--",
                                  label="Ground Truth", linewidth=1.2)
                    ax_f.semilogy(f_pred,   psd_pred,   color="#d62728",
                                  label="Prediction",   linewidth=1.2)
                    ax_f.set_xlabel("Normalised Frequency")
                    ax_f.set_ylabel("PSD")
                    ax_f.set_title(f"Power Spectral Density (ch {ch_i})")
                    ax_f.legend(frameon=False, loc="upper right")
                    ax_f.grid(True, alpha=0.35, which="both")
                    ax_f.spines["top"].set_visible(False)
                    ax_f.spines["right"].set_visible(False)

                    fig.suptitle(
                        f"Inference @ Epoch {epoch + 1}  |  Sample {sample_num}  |  Channel {ch_i}",
                        y=1.01,
                    )
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(fig_dir, f"infer_epoch_{epoch:04d}_sample_{sample_num:04d}_ch{ch_i}.png"),
                        dpi=300, bbox_inches="tight",
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

    train_ds = InverseH5Dataset(train_h5)
    val_ds   = InverseH5Dataset(val_h5)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True,
    ) if len(val_ds) > 0 else None

    in_channels  = train_ds[0]["y"].shape[-1]   # n_sensors
    out_channels = train_ds[0]["u"].shape[-1]   # n_irr_dofs

    print("=" * 80)
    print(f"Device       : {device}")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"in_channels={in_channels}, out_channels={out_channels}")
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
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)
            target = batch["u"].to(device, non_blocking=True).transpose(1, 2)
            pred   = model(inp)
            loss   = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4e}")

        avg_loss = total_loss / len(train_loader)
        val_loss = 0.0
        if val_loader is not None:
            val_loss = test_step(model, val_loader, epoch, device, args.fig_dir, args.vis_every)

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
        inference_and_plot(model, val_loader, device, args.fig_dir, epoch=args.epochs - 1)
        print(f"Figures saved to: {args.fig_dir}")


if __name__ == "__main__":
    main()
