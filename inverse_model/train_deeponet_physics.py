'''
Author: 2099193635 2099193635@qq.com
Date: 2026-04-24
FilePath: /VTCM_PYTHON/inverse_model/train_deeponet_physics.py
Description: DeepONet 训练脚本（含物理信息嵌入 v2）
  在 train_deeponet.py 基础上增加了 PhysicsLayerV3，支持两类物理约束：
    dyn  ─ 车辆-轨道动力学方程 ODE 残差（时域）
    psd  ─ 轨道不平顺空间谱密度约束
    both ─ spectral + frf
'''
"""
DeepONet 训练脚本（含物理信息嵌入 v2）
从 inverse_dataset_gen 导出的 HDF5 数据集训练 VTCMDeepONet 模型。
输入：车辆传感器加速度 y [T, n_sensors]
输出：轨道不平顺激励 u [T, n_irr_dofs]

物理损失模式（--physics_mode）：
  none  ─ 纯数据驱动（与 train_deeponet.py 等价）
  dyn   ─ 车辆-轨道耦合动力学方程 ODE 残差（时域模态叠加）
  psd   ─ 轨道不平顺空间功率谱密度约束（对数域 MSE）
  both  ─ spectral + frf 同时开启
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
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils.checkpoint import save_checkpoint
from differentiable_physics_v3 import PhysicsLayerV3     # v3 物理层


# ── DeepONet 模型 ─────────────────────────────────────────────────────────────
class VTCMDeepONet(torch.nn.Module):
    """
    1-D DeepONet 改进版：branch 保留 T 维，trunk 生成与 branch 等长的时域坐标编码。

    融合策略：拼接（Concatenation）而非 Hadamard 乘法
        原因：Hadamard 乘法在初始化阶段容易出现梯度湮灭（branch≈0 或 trunk≈0 时，
              对方的梯度也被归零，两侧互相锁死，导致训练无效）。
              Concat+Linear 保持加法梯度路径，训练更稳定。

    Branch : FNO  y [B, n_sensors, T]  ->  [B, latent_dim, T]    保留时序
    Trunk  : MLP  t_norm [B, T, 1]    ->  [B, T, latent_dim]    时域坐标编码
               （t_norm = linspace(0,1,T)，在 forward 内自动生成，不依赖 x_query）
    Fuse   : cat([branch, trunk], -1)  ->  [B, T, 2*latent_dim]  拼接
    Proj   : Linear(2*latent, n_irr)   ->  [B, T, n_irr_dofs]
    Output : permute                   ->  [B, n_irr_dofs, T]
    """

    def __init__(
        self,
        branch_net: torch.nn.Module,
        trunk_net: torch.nn.Module,
        n_irr_dofs: int = 1,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.branch_net  = branch_net
        self.trunk_net   = trunk_net
        # 拼接后维度为 2*latent_dim
        self.output_proj = torch.nn.Linear(2 * latent_dim, n_irr_dofs)
        # 输出层小初始化，避免初始输出过大导致 loss 震荡
        torch.nn.init.normal_(self.output_proj.weight, std=0.01)
        torch.nn.init.zeros_(self.output_proj.bias)
        self.n_irr_dofs  = n_irr_dofs
        self.latent_dim  = latent_dim

    def forward(self, y: torch.Tensor, x_query: torch.Tensor = None) -> torch.Tensor:
        """
        y       : [B, n_sensors, T]  (channel-first)
        x_query : 保留参数兼容性，实际不使用；trunk 内部生成 linspace(0,1,T)
        returns : [B, n_irr_dofs, T]
        """
        B, _, T = y.shape

        # ── Branch：FNO 保留 T 维 ──────────────────────────────────────────
        branch_out = self.branch_net(y)            # [B, latent_dim, T]
        branch_out = branch_out.permute(0, 2, 1)   # [B, T, latent_dim]

        # ── Trunk：MLP 编码归一化时间坐标（内部生成，不依赖 x_query）─────────
        # 使用时域坐标而非空间坐标，语义一致
        t_norm    = torch.linspace(0, 1, T, device=y.device, dtype=y.dtype)  # [T]
        t_in      = t_norm.view(1, T, 1).expand(B, T, 1)                    # [B, T, 1]
        trunk_out = self.trunk_net(t_in)                                      # [B, T, latent_dim]

        # ── 拼接融合 + 线性投影（替代 Hadamard，梯度流更稳定）─────────────────
        fused = torch.cat([branch_out, trunk_out], dim=-1)  # [B, T, 2*latent_dim]
        out   = self.output_proj(fused)                     # [B, T, n_irr_dofs]
        return out.permute(0, 2, 1)                         # [B, n_irr_dofs, T]


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
    p = argparse.ArgumentParser(description="DeepONet Training for VTCM Inverse Problem (with Physics)")
    p.add_argument("--epochs",             type=int,   default=200)
    p.add_argument("--batch_size",         type=int,   default=16)
    p.add_argument("--lr",                 type=float, default=3e-3)
    p.add_argument("--gamma",              type=float, default=0.99948708,
                   help="ExponentialLR 衰减系数")
    p.add_argument("--latent_channels",    type=int,   default=128)
    p.add_argument("--num_fno_layers",     type=int,   default=8)
    p.add_argument("--num_fno_modes",      type=int,   default=24)
    p.add_argument("--decoder_layers",     type=int,   default=3)
    p.add_argument("--decoder_layer_size", type=int,   default=64)
    p.add_argument("--padding",            type=int,   default=9)
    p.add_argument("--ckpt_dir",           type=str,   default=None,
                   help="Checkpoint 保存目录，默认根据 physics_mode 自动命名")
    p.add_argument("--fig_dir",            type=str,   default=None,
                   help="图片保存目录，默认根据 physics_mode 自动命名")
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
    # ── 物理损失相关（v3）────────────────────────────────────────────────────
    p.add_argument("--physics_mode",     type=str,   default="both",
                   choices=["none", "spectral", "frf", "both"],
                   help=("物理模式: "
                         "none/spectral=频谱加权数据损失（改善高频细节）; "
                         "frf/both=频谱损失+FRF正向一致性物理约束"))
    p.add_argument("--lambda_frf",       type=float, default=1.0,
                   help="FRF 正向一致性损失权重（frf/both 模式有效）")
    p.add_argument("--n_modal",          type=int,   default=8,
                   help="模态数（条件向量中 nat_freqs / damp_ratios 的长度）")
    p.add_argument("--dx",               type=float, default=0.25,
                   help="空间步长 [m]")
    p.add_argument("--dt",               type=float, default=0.005,
                   help="时间步长 [s]")
    # ── 频谱加权损失参数 ──────────────────────────────────────────────────────
    p.add_argument("--spectral_beta",    type=float, default=0.5,
                   help="频谱项权重（相对时域 MSE）")
    p.add_argument("--spectral_gamma",   type=float, default=5.0,
                   help="高频强调因子（越大越关注高频细节）")
    p.add_argument("--spectral_alpha",   type=float, default=1.0,
                   help="频率权重幂次（1=线性，2=二次）")
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
            inp     = batch["y"].to(device, non_blocking=True).transpose(1, 2)   # [B, n_s, T]
            x_query = batch["x_query"].to(device, non_blocking=True)              # [B, L]
            target  = batch["u"].to(device, non_blocking=True).permute(0, 2, 1)   # [B, n_dir, L]
            pred    = model(inp, x_query)
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

        for ch_i in range(n_ch):
            sig_pred   = viz_pred[ch_i]
            sig_target = viz_target[ch_i]

            from scipy.signal import welch
            nperseg = min(256, T // 4)
            f_pred,   psd_pred   = welch(sig_pred,   nperseg=nperseg)
            f_target, psd_target = welch(sig_target, nperseg=nperseg)

            fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(14, 4))

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
            pred    = model(inp, x_query)                                          # [B, n_dir, T]

            for i in range(pred.shape[0]):
                n_ch   = pred.shape[1]
                T      = pred.shape[2]
                t_axis = np.arange(T)

                for ch_i in range(n_ch):
                    sig_pred   = pred[i, ch_i].cpu().numpy()
                    sig_target = output[i, ch_i].cpu().numpy()

                    nperseg = min(256, T // 4)
                    f_pred,   psd_pred   = welch(sig_pred,   nperseg=nperseg)
                    f_target, psd_target = welch(sig_target, nperseg=nperseg)

                    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(14, 4))

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

    # ── 根据 physics_mode 自动设置保存目录 ───────────────────────────────────
    mode_tag = args.physics_mode  # none / spectral / frf / both
    if args.ckpt_dir is None:
        args.ckpt_dir = str(_DIR / f"checkpoints_deeponet_{mode_tag}")
    if args.fig_dir is None:
        args.fig_dir  = str(_DIR / f"figures_deeponet_{mode_tag}")

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
    print(f"Physics mode : {args.physics_mode}")
    print("=" * 80)

    # DeepONet 超参
    latent_dim = args.latent_channels
    n_irr_dofs = out_channels

    model_branch = FNO(
        in_channels=in_channels,
        out_channels=latent_dim,
        decoder_layers=args.decoder_layers,
        decoder_layer_size=args.decoder_layer_size,
        dimension=1,
        latent_channels=args.latent_channels,
        num_fno_layers=args.num_fno_layers,
        num_fno_modes=args.num_fno_modes,
        padding=args.padding,
    )

    model_trunk = FullyConnected(
        in_features=1,
        out_features=latent_dim,
        layer_size=128,
        num_layers=4,
    )

    model = VTCMDeepONet(
        branch_net=model_branch,
        trunk_net=model_trunk,
        n_irr_dofs=n_irr_dofs,
        latent_dim=latent_dim,
    ).to(device)

    # ── 物理层 v3（始终实例化，spectral 模式不增加额外参数）────────────────────
    physics_layer = PhysicsLayerV3(
        mode=args.physics_mode,
        n_modal=args.n_modal,
        n_sensors=in_channels,
        dx=args.dx,
        dt=args.dt,
        spectral_beta=args.spectral_beta,
        spectral_gamma=args.spectral_gamma,
        spectral_alpha=args.spectral_alpha,
    ).to(device)
    all_params = list(model.parameters()) + list(physics_layer.parameters())
    print(f"PhysicsLayerV3 mode={args.physics_mode}, "
          f"use_frf={physics_layer.use_frf}, "
          f"learnable params: {sum(p.numel() for p in physics_layer.parameters())}")

    optimizer = torch.optim.Adam(
        all_params, betas=(0.9, 0.999), lr=args.lr, weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    n_modal = args.n_modal

    for epoch in range(args.epochs):
        model.train()
        if physics_layer is not None:
            physics_layer.train()
        total_loss      = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for batch in pbar:
            optimizer.zero_grad()
            inp     = batch["y"].to(device, non_blocking=True).transpose(1, 2)   # [B, n_s, T]
            x_query = batch["x_query"].to(device, non_blocking=True)              # [B, L]
            target  = batch["u"].to(device, non_blocking=True).permute(0, 2, 1)   # [B, n_dir, L]
            pred    = model(inp, x_query)                                           # [B, n_dir, T]

            # 频谱加权数据损失（替代纯 MSE，改善高频细节学习）
            loss_data = physics_layer.compute_data_loss(pred, target)

            if physics_layer.use_frf:
                # 从条件向量 c 中解析出模态参数
                # c 结构：[nat_freqs(n_modal) | damp_ratios(n_modal) | vx_mps]
                c           = batch["c"].to(device, non_blocking=True)             # [B, 2*n_modal+1]
                nat_freqs   = c[:, :n_modal]                                       # [B, n_modal]
                damp_ratios = c[:, n_modal:2 * n_modal]                            # [B, n_modal]
                vx_mps      = c[:, -1]                                             # [B]

                # 取第一个不平顺通道作为空间不平顺 z_pred [B, T]
                z_pred   = pred[:, 0, :].contiguous()                             # [B, T]
                a_c_true = batch["y"].to(device, non_blocking=True)               # [B, T, n_sensors]

                phys_loss, phys_metrics = physics_layer(
                    z_pred      = z_pred,
                    a_c_true    = a_c_true,
                    nat_freqs   = nat_freqs,
                    damp_ratios = damp_ratios,
                    vx_mps      = vx_mps,
                    lambda_frf  = args.lambda_frf,
                )
                loss = loss_data + phys_loss
                total_phys_loss += phys_loss.item()
                pbar.set_postfix(
                    loss=f"{loss.item():.4e}",
                    data=f"{loss_data.item():.4e}",
                    frf=f"{phys_metrics['l_frf']:.3e}",
                )
            else:
                loss = loss_data
                pbar.set_postfix(loss=f"{loss.item():.4e}")

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss      += loss.item()
            total_data_loss += loss_data.item()

        avg_loss = total_loss / len(train_loader)
        avg_data = total_data_loss / len(train_loader)
        avg_phys = total_phys_loss / len(train_loader)

        val_loss = 0.0
        if val_loader is not None:
            val_loss = test_step(model, val_loader, epoch, device, args.fig_dir, args.vis_every)

        if physics_layer.use_frf:
            print(
                f"[Epoch {epoch + 1:4d}/{args.epochs}] "
                f"train={avg_loss:.4e}  data={avg_data:.4e}  frf={avg_phys:.4e}  "
                f"val={val_loss:.4e}  lr={optimizer.param_groups[0]['lr']:.3e}"
            )
        else:
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
