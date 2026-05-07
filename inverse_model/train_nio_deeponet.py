"""
NIO 框架 DeepONet 训练脚本 —— VTCM 逆问题（序列预测版）
========================================================
本问题是序列到序列的逆问题，不同于 NIO 原版预测固定位置的值，
需要预测一整段时序 u(t)，因此 Branch 必须保留时间维 T。

  Branch  : 1D CNN 编码器  y [B, n_s, T]  ──→  [B, n_basis, T]  (时变基函数系数)
  Trunk   : FeedForwardNN  t [T, 1]        ──→  [T, n_basis]      (时间基函数值)
  Fusion  : 逐时刻内积  out[b,t] = Σ_k branch[b,k,t] · trunk[t,k]  →  [B, T]
  Refine  : FNO1d（可选，NIO 标准精修层）                          ──→  [B, T]

与 NIO 原版 SNOConvRad 的关键差异：
  - NIO：Branch GlobalAvgPool 压缩到 [B, n_basis]，系数全局固定（适合固定位置预测）
  - 本版：Branch 保留 T 维 → 系数随时间变化（适合序列区段预测）

用法示例::

    python train_nio_deeponet.py \\
        --n_basis 128 --branch_hidden 128 \\
        --trunk_neurons 128 --trunk_layers 4 \\
        --fno_modes 16 --fno_width 64 --fno_layers 4 \\
        --epochs 200 --batch_size 16 --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ── 路径配置 ──────────────────────────────────────────────────────────────────
_DIR  = Path(__file__).parent
_ROOT = _DIR.parent
_NIO  = str(_ROOT / "NIO")

# physicsnemo 优先使用工作区本地版本
for _p in [str(_ROOT / "physicsnemo"), str(_DIR), _NIO]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# NIO 核心模块
from DeepONetModules import FeedForwardNN  # noqa: E402
from FNOModules import FNO1d               # noqa: E402

from physicsnemo.utils.checkpoint import save_checkpoint       # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Branch 编码器（对标 NIO/Baselines.py 中的 EncoderRad）
# ══════════════════════════════════════════════════════════════════════════════

class BranchEncoder1d(nn.Module):
    """
    1D CNN 编码器（序列版）：
        y [B, n_sensors, T]  →  [B, n_basis, T]

    保留时间维 T，每个时刻独立输出 n_basis 维系数，
    与 Trunk 做逐时刻内积，适合序列区段预测。
    （与 NIO 原版不同：NIO 做 GlobalAvgPool 压缩到 [B, n_basis]，仅适合固定位置预测）
    """

    def __init__(self, in_channels: int, n_basis: int, hidden_channels: int = 128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, n_basis, kernel_size=3, padding=1),
        )
        self.n_basis = n_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_sensors, T] → [B, n_basis, T]  (保留时间维)"""
        return self.convs(x)      # [B, n_basis, T]


# ══════════════════════════════════════════════════════════════════════════════
# 单通道 NIO 算子（对标 SNOConvRad）
# ══════════════════════════════════════════════════════════════════════════════

class _SingleChannelNIONet(nn.Module):
    """
    序列版 NIO 单通道算子：

        Branch 系数（时变）：branch(y) → [B, n_basis, T]
        Trunk  基函数值  ：trunk(t)   → [T, n_basis]
        逐时刻内积       ：out[b,t] = Σ_k coef[b,k,t] · basis[t,k]  →  [B, T]
        可选 FNO1d 精修

    与 NIO SNOConvRad 的差异：
        NIO  用 DeepOnetNoBiasOrg(branch[B,p] × trunk[T,p]) → [B,T]（全局系数）
        本版 用 einsum('bkt,tk->bt') 实现逐时刻加权（时变系数）
    """

    def __init__(
        self,
        trunk_arch: dict,
        n_basis: int,
        fno_arch: dict | None,
        device: torch.device,
    ):
        super().__init__()
        self.n_basis = n_basis
        self.trunk   = FeedForwardNN(1, n_basis, trunk_arch)
        self.b0      = nn.Parameter(torch.tensor(0.0))   # 可学习偏置（对标 DeepOnetNoBiasOrg）
        self.use_fno = (fno_arch is not None) and (fno_arch.get("n_layers", 0) > 0)
        if self.use_fno:
            self.fno = FNO1d(fno_arch, device=str(device), nfun=1)

    def forward(self, branch_coef: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        """
        branch_coef : [B, n_basis, T]  由外部 Branch 编码器产生的时变系数
        t_grid      : [T, 1]           归一化时间坐标（共享，无 batch dim）
        return      : [B, T]
        """
        T = branch_coef.shape[-1]
        basis = self.trunk(t_grid)                            # [T, n_basis]
        # 逐时刻内积：out[b,t] = sum_k coef[b,k,t] * basis[t,k]
        x = torch.einsum('bkt,tk->bt', branch_coef, basis)   # [B, T]
        x = (x + self.b0) / (self.n_basis ** 0.5)            # 归一化（与 NIO 一致）

        if self.use_fno:
            B = x.shape[0]
            x = x.unsqueeze(-1)                                       # [B, T, 1]
            grid_expand = t_grid.unsqueeze(0).expand(B, T, 1)         # [B, T, 1]
            x_in = torch.cat([x, grid_expand], dim=-1)                # [B, T, 2]
            x = self.fno(x_in)                                        # [B, T]

        return x                                              # [B, T]


# ══════════════════════════════════════════════════════════════════════════════
# 顶层模型
# ══════════════════════════════════════════════════════════════════════════════

class VTCMNIONet(nn.Module):
    """
    VTCM 逆问题 NIO 框架实现。

    对 n_irr_dofs 个输出通道，共享同一个 BranchEncoder1d，
    每个通道独立配置 Trunk + DeepONet + FNO1d。
    """

    def __init__(
        self,
        n_sensors: int,
        n_irr_dofs: int,
        n_basis: int,
        branch_hidden: int,
        trunk_arch: dict,
        fno_arch: dict | None,
        device: torch.device,
    ):
        super().__init__()
        self.n_irr_dofs = n_irr_dofs

        # 共享 Branch 编码器（所有通道共用）
        self.branch = BranchEncoder1d(n_sensors, n_basis, branch_hidden)

        # 每通道独立的 Trunk + 逐时刻内积 + FNO
        self.channels = nn.ModuleList([
            _SingleChannelNIONet(
                trunk_arch=trunk_arch,
                n_basis=n_basis,
                fno_arch=fno_arch,
                device=device,
            )
            for _ in range(n_irr_dofs)
        ])

    def forward(self, y: torch.Tensor, x_query: torch.Tensor = None) -> torch.Tensor:
        """
        y       : [B, n_sensors, T]  channel-first
        x_query : 保留兼容性，trunk 内部生成 linspace(0,1,T)
        return  : [B, n_irr_dofs, T]
        """
        B, _, T = y.shape
        t_grid = torch.linspace(0, 1, T, device=y.device, dtype=y.dtype).unsqueeze(-1)  # [T, 1]

        # Branch 只调用一次，所有通道共享时变系数
        branch_coef = self.branch(y)                              # [B, n_basis, T]

        outs = [ch(branch_coef, t_grid) for ch in self.channels]  # n_irr_dofs × [B, T]
        return torch.stack(outs, dim=1)                           # [B, n_irr_dofs, T]


# ══════════════════════════════════════════════════════════════════════════════
# 数据集（与 train_deeponet.py 完全相同）
# ══════════════════════════════════════════════════════════════════════════════

class InverseH5Dataset(Dataset):
    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        with h5py.File(self.h5_path, "r") as f:
            self.y        = np.asarray(f["y"],       dtype=np.float32)
            self.u        = np.asarray(f["u"],       dtype=np.float32)
            self.c        = np.asarray(f["c"],       dtype=np.float32)
            self.x_query  = np.asarray(f["x_query"], dtype=np.float32)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "y":       torch.from_numpy(self.y[idx]),
            "u":       torch.from_numpy(self.u[idx]),
            "c":       torch.from_numpy(self.c[idx]),
            "x_query": torch.from_numpy(self.x_query[idx]),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VTCM NIO-DeepONet 训练脚本")
    # 训练超参
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gamma",        type=float, default=0.99948708,
                   help="ExponentialLR 衰减系数")
    p.add_argument("--patience",     type=int,   default=-1,
                   help="Early stopping patience（-1 禁用）")
    # Branch 超参
    p.add_argument("--n_basis",        type=int, default=64,
                   help="Branch/Trunk 共享基函数维数")
    p.add_argument("--branch_hidden",  type=int, default=64,
                   help="Branch CNN 隐藏通道数")
    # Trunk 超参
    p.add_argument("--trunk_neurons",  type=int, default=64)
    p.add_argument("--trunk_layers",   type=int, default=3)
    p.add_argument("--trunk_act",      type=str, default="tanh")
    p.add_argument("--trunk_dropout",  type=float, default=0.0)
    p.add_argument("--trunk_retrain",  type=int, default=42)
    # FNO1d 精修层超参
    p.add_argument("--fno_modes",   type=int, default=16)
    p.add_argument("--fno_width",   type=int, default=64)
    p.add_argument("--fno_layers",  type=int, default=4,
                   help="FNO1d 精修层数（0 = 关闭）")
    p.add_argument("--fno_retrain", type=int, default=42)
    # 路径
    p.add_argument("--ckpt_dir",     type=str,
                   default=str(_DIR / "checkpoints_nio_deeponet"))
    p.add_argument("--fig_dir",      type=str,
                   default=str(_DIR / "figures_nio_deeponet"))
    p.add_argument("--vis_every",    type=int, default=10)
    p.add_argument("--dataset_dir",  type=str,
                   default=str(_ROOT / "datasets" / "VTCM_inverse"))
    p.add_argument("--train-full-seq", dest="train_full_seq",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-full-seq",   dest="val_full_seq",
                   action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


# ══════════════════════════════════════════════════════════════════════════════
# 验证 & 可视化
# ══════════════════════════════════════════════════════════════════════════════

def _setup_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 11,   "axes.titlesize": 12,
        "legend.fontsize": 9,   "lines.linewidth": 1.4,
        "figure.dpi": 150,
    })


def _plot_timeseries_psd(sig_pred, sig_target, ch_i, epoch, fig_dir, prefix="val", sample_num=None):
    from scipy.signal import welch
    T = len(sig_pred)
    nperseg = min(256, T // 4)
    f_p, psd_p = welch(sig_pred,   nperseg=nperseg)
    f_t, psd_t = welch(sig_target, nperseg=nperseg)

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(14, 4))
    ax_t.plot(np.arange(T), sig_target, color="#1f77b4", linestyle="--",
              label="Ground Truth", linewidth=1.2)
    ax_t.plot(np.arange(T), sig_pred,   color="#d62728",
              label="Prediction",  linewidth=1.2)
    ax_t.set_xlabel("Time Step"); ax_t.set_ylabel("Amplitude")
    ax_t.set_title(f"Time Series (ch {ch_i})")
    ax_t.legend(frameon=False); ax_t.grid(True, alpha=0.35)
    ax_t.spines["top"].set_visible(False); ax_t.spines["right"].set_visible(False)

    ax_f.semilogy(f_t, psd_t, color="#1f77b4", linestyle="--",
                  label="Ground Truth", linewidth=1.2)
    ax_f.semilogy(f_p, psd_p, color="#d62728",
                  label="Prediction",  linewidth=1.2)
    ax_f.set_xlabel("Normalised Frequency"); ax_f.set_ylabel("PSD")
    ax_f.set_title(f"Power Spectral Density (ch {ch_i})")
    ax_f.legend(frameon=False); ax_f.grid(True, alpha=0.35, which="both")
    ax_f.spines["top"].set_visible(False); ax_f.spines["right"].set_visible(False)

    title = f"Epoch {epoch + 1}  |  ch {ch_i}"
    if sample_num is not None:
        title = f"Sample {sample_num}  |  " + title
    fig.suptitle(title, y=1.01)
    fig.tight_layout()

    fname = (
        f"infer_epoch_{epoch:04d}_sample_{sample_num:04d}_ch{ch_i}.png"
        if sample_num is not None
        else f"{prefix}_epoch_{epoch:04d}_ch{ch_i}.png"
    )
    fig.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    device: torch.device,
    fig_dir: str,
    vis_every: int,
) -> float:
    model.eval()
    loss_sum, n_batch = 0.0, 0
    viz_pred = viz_target = None

    with torch.no_grad():
        for batch in dataloader:
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)   # [B, n_s, T]
            x_q    = batch["x_query"].to(device, non_blocking=True)
            target = batch["u"].to(device, non_blocking=True).permute(0, 2, 1)  # [B, n_dir, L]
            pred   = model(inp, x_q)
            loss_sum += F.mse_loss(pred, target).item()
            n_batch  += 1
            if viz_pred is None:
                viz_pred   = pred[0].detach().cpu().numpy()
                viz_target = target[0].detach().cpu().numpy()

    avg_loss = loss_sum / max(n_batch, 1)

    if viz_pred is not None and (epoch % vis_every == 0):
        os.makedirs(fig_dir, exist_ok=True)
        _setup_plot_style()
        for ch_i in range(viz_pred.shape[0]):
            _plot_timeseries_psd(viz_pred[ch_i], viz_target[ch_i],
                                 ch_i, epoch, fig_dir, prefix="val")

    return avg_loss


def inference_and_plot(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    fig_dir: str,
    epoch: int,
) -> None:
    model.eval()
    os.makedirs(fig_dir, exist_ok=True)
    _setup_plot_style()
    sample_num = 0
    with torch.no_grad():
        for batch in dataloader:
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)
            x_q    = batch["x_query"].to(device, non_blocking=True)
            target = batch["u"].to(device, non_blocking=True).permute(0, 2, 1)
            pred   = model(inp, x_q)
            for i in range(pred.shape[0]):
                for ch_i in range(pred.shape[1]):
                    _plot_timeseries_psd(
                        pred[i, ch_i].cpu().numpy(),
                        target[i, ch_i].cpu().numpy(),
                        ch_i, epoch, fig_dir, sample_num=sample_num,
                    )
                sample_num += 1


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

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

    n_sensors  = train_ds[0]["y"].shape[-1]    # [T, n_s] → n_s
    n_irr_dofs = train_ds[0]["u"].shape[-1]    # [L, n_dir] → n_dir

    print("=" * 80)
    print(f"Device         : {device}")
    print(f"Train samples  : {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"n_sensors={n_sensors},  n_irr_dofs={n_irr_dofs}")
    print(f"Architecture   : Branch(1D-CNN) → DeepONet(dot-product) → FNO1d(refine)")
    print(f"n_basis={args.n_basis}, branch_hidden={args.branch_hidden}, "
          f"fno_layers={args.fno_layers}")
    print("=" * 80)

    # ── Trunk 配置（对应 FeedForwardNN network_architecture 字典）──
    trunk_arch = {
        "n_hidden_layers": args.trunk_layers,
        "neurons":         args.trunk_neurons,
        "act_string":      args.trunk_act,
        "retrain":         args.trunk_retrain,
        "dropout_rate":    args.trunk_dropout,
    }

    # ── FNO1d 配置（对应 FNO1d fno_architecture 字典）──
    fno_arch = None
    if args.fno_layers > 0:
        fno_arch = {
            "modes":      args.fno_modes,
            "width":      args.fno_width,
            "n_layers":   args.fno_layers,
            "retrain_fno": args.fno_retrain,
        }

    model = VTCMNIONet(
        n_sensors    = n_sensors,
        n_irr_dofs   = n_irr_dofs,
        n_basis      = args.n_basis,
        branch_hidden= args.branch_hidden,
        trunk_arch   = trunk_arch,
        fno_arch     = fno_arch,
        device       = device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.999),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    best_val  = float("inf")
    no_improv = 0

    for epoch in range(args.epochs):
        # ── 训练 ──
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)   # [B, n_s, T]
            x_q    = batch["x_query"].to(device, non_blocking=True)
            target = batch["u"].to(device, non_blocking=True).permute(0, 2, 1)  # [B, n_dir, L]
            pred   = model(inp, x_q)
            loss   = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4e}")

        avg_loss = total_loss / len(train_loader)

        # ── 验证 ──
        val_loss = 0.0
        if val_loader is not None:
            val_loss = test_step(model, val_loader, epoch, device,
                                 args.fig_dir, args.vis_every)

        print(
            f"[Epoch {epoch + 1:4d}/{args.epochs}] "
            f"train={avg_loss:.4e}  val={val_loss:.4e}  "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        # ── 检查点 ──
        save_checkpoint(
            args.ckpt_dir, models=model,
            optimizer=optimizer, scheduler=scheduler, epoch=epoch,
        )

        # ── Early stopping ──
        if args.patience > 0 and val_loader is not None:
            if val_loss < best_val:
                best_val  = val_loss
                no_improv = 0
                torch.save(model.state_dict(),
                           os.path.join(args.ckpt_dir, "best_model.pth"))
                print(f"  ✓ Best val loss: {best_val:.4e}  (saved best_model.pth)")
            else:
                no_improv += 1
                if no_improv >= args.patience:
                    print(f"Early stopping at epoch {epoch + 1} "
                          f"(no improvement for {args.patience} epochs)")
                    break

    # ── 训练结束后全量推理可视化 ──
    if val_loader is not None:
        print("Generating full inference plots …")
        inference_and_plot(model, val_loader, device, args.fig_dir, epoch=args.epochs - 1)
        print(f"Figures saved to: {args.fig_dir}")


if __name__ == "__main__":
    main()
