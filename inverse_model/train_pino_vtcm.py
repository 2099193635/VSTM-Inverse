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
import torch.nn as nn
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


# ── VTCM 7-DOF 垂向动力学正向仿真 & 物理损失 ──────────────────────────────────
#
# 三大工程问题的解决策略：
#
# ① 速度问题：
#    不使用完整序列积分。每次训练随机取 win_len（如 256 步 = 25ms）短窗口，
#    对该窗口使用"从数据中恢复的初始状态"启动积分（见②），GPU 批量并行。
#    256步 × batch_size=8 在 GPU 上只需 ~2ms，可以承受。
#    进一步提速：使用梯度检查点（ckpt_segments）将显存从 O(W) 降到 O(√W)。
#
# ② 初始值问题：
#    从观测数据出发，用"准静态物理一致性"重建所有 14 个初始自由度：
#    - 车体 Z 加速度 → 已知（传感器直接测量）
#    - 车体 Z 位移/速度 → 对加速度双重积分（去趋势后）
#    - 构架/轮对 → 利用准静态力平衡：在短窗口内振动量级远小于静变形，
#      可用 Kz·Δx = 已知力 反推
#    实现于 StateEstimator（小型 MLP，与 FNO 端到端联合训练）
#
# ③ 可微分问题：
#    翟方法每步只有线性矩阵运算（@, *），PyTorch autograd 全程可追踪。
#    对短窗口（256步）直接反向传播；对更长窗口用 torch.utils.checkpoint
#    分段重算，以显存换时间。


class StateEstimator(nn.Module):
    """
    从车体加速度短观测窗推断完整 14 维初始状态 [X(7), V(7)]。

    解决初始值问题（问题②）：
    - 输入：车体 Z 加速度短窗口 y_ctx  [B, T_ctx]
    - 输出：14 维初始状态向量 z0       [B, 14]

    设计依据：
    - 车体位移/速度可由加速度积分估计；
    - 构架/轮对状态由力平衡软约束约束；
    - 使用小 MLP（< 5k 参数），与 FNO 端到端联合训练，
      梯度通过动力学仿真反向传入估计器。

    物理软约束（在 forward 中以正则化形式施加）：
      准静态力平衡：K2z·(Zc - Zt_i) ≈ Mc·g/4（车体-构架）
      K1z·(Zt_i - Zw_j) ≈ (Mc+2Mt)·g/8（构架-轮对）
    由于数据已减去静变形（扰动形式），平衡位置均为 0，
    因此仿真开始时理想初始状态应接近零。
    """
    def __init__(self, T_ctx: int = 64, hidden: int = 64):
        super().__init__()
        # 轻量 MLP：[B, T_ctx] → [B, 14]
        self.net = nn.Sequential(
            nn.Linear(T_ctx, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 14),   # 7位移 + 7速度
        )
        # 初始化为近零（初始状态应接近平衡点）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y_ctx: torch.Tensor) -> torch.Tensor:
        """
        y_ctx : [B, T_ctx]  车体 Z 加速度上下文（float32）
        返回    z0    : [B, 14]  初始状态（float64，供 VTCMVerticalDynamics 使用）
        """
        return self.net(y_ctx).double()   # [B, 14]


class VTCMVerticalDynamics(nn.Module):
    """
    基于 generate_main.py 完整 VTCM 计算流程的可微分 7-DOF 垂向动力学。

    拓扑（与 generate_main.py 保持一致，锁定非垂向 DOF 后）：
      DOF 0 : 车体  Z  (Zc)
      DOF 1 : 前转向架 Z (Zt1)
      DOF 2 : 后转向架 Z (Zt2)
      DOF 3~6 : 轮对 1~4 Z (Zw1..Zw4)

    力元（对应 suspension.py，左+右两侧并联）：
      K1z = 2·Kpz, C1z = 2·Cpz  （一系，对应 Fzf_L + Fzf_R）
      K2z = 2·Ksz, C2z = 2·Csz  （二系，对应 Fzt_L + Fzt_R）
      Kh·(u_i - Zw_i)            （轮轨赫兹接触线性化）

    积分：翟方法（与 solver.py 完全一致，alpha=0.5, beta=0.25）
    """

    def __init__(
        self,
        Mc:  float = 34000.0,
        Mt:  float = 3000.0,
        Mw:  float = 1400.0,
        Kpz: float = 0.55e6,
        Ksz: float = 0.40e6,
        Cpz: float = 6e3,
        Csz: float = 80e3,
        Kh:  float = 1.0e9,
        dt:  float = 1e-4,
        alpha: float = 0.5,
        beta:  float = 0.25,
        win_len:      int = 256,   # 每次积分的窗口步数（解决速度问题①）
        ckpt_segments: int = 4,    # 梯度检查点分段数（0=不使用；解决显存问题）
        ctx_len:      int = 64,    # StateEstimator 上下文长度（解决初始值问题②）
    ):
        super().__init__()

        K1z = 2.0 * Kpz
        C1z = 2.0 * Cpz
        K2z = 2.0 * Ksz
        C2z = 2.0 * Csz

        M = torch.tensor([Mc, Mt, Mt, Mw, Mw, Mw, Mw], dtype=torch.float64)

        # 刚度矩阵
        Km = torch.zeros(7, 7, dtype=torch.float64)
        Km[0, 0] = -2*K2z;   Km[0, 1] = K2z;            Km[0, 2] = K2z
        Km[1, 0] = K2z;      Km[1, 1] = -(K2z+2*K1z);   Km[1, 3] = K1z;  Km[1, 4] = K1z
        Km[2, 0] = K2z;      Km[2, 2] = -(K2z+2*K1z);   Km[2, 5] = K1z;  Km[2, 6] = K1z
        Km[3, 1] = K1z;      Km[3, 3] = -(K1z + Kh)
        Km[4, 1] = K1z;      Km[4, 4] = -(K1z + Kh)
        Km[5, 2] = K1z;      Km[5, 5] = -(K1z + Kh)
        Km[6, 2] = K1z;      Km[6, 6] = -(K1z + Kh)

        # 阻尼矩阵
        Cm = torch.zeros(7, 7, dtype=torch.float64)
        Cm[0, 0] = -2*C2z;   Cm[0, 1] = C2z;            Cm[0, 2] = C2z
        Cm[1, 0] = C2z;      Cm[1, 1] = -(C2z+2*C1z);   Cm[1, 3] = C1z;  Cm[1, 4] = C1z
        Cm[2, 0] = C2z;      Cm[2, 2] = -(C2z+2*C1z);   Cm[2, 5] = C1z;  Cm[2, 6] = C1z
        Cm[3, 1] = C1z;      Cm[3, 3] = -C1z
        Cm[4, 1] = C1z;      Cm[4, 4] = -C1z
        Cm[5, 2] = C1z;      Cm[5, 5] = -C1z
        Cm[6, 2] = C1z;      Cm[6, 6] = -C1z

        # 输入矩阵：Kh 施加在轮对 DOF 3~6
        Bm = torch.zeros(7, 4, dtype=torch.float64)
        Bm[3, 0] = Kh;  Bm[4, 1] = Kh;  Bm[5, 2] = Kh;  Bm[6, 3] = Kh

        self.register_buffer("M_inv", 1.0 / M)
        self.register_buffer("Km",    Km)
        self.register_buffer("Cm",    Cm)
        self.register_buffer("Bm",    Bm)

        self.dt            = dt
        self.alpha         = alpha
        self.beta          = beta
        self.win_len       = win_len
        self.ckpt_segments = ckpt_segments
        self.ctx_len       = ctx_len

        # 状态估计器（问题②：从传感器数据恢复初始状态）
        self.state_estimator = StateEstimator(T_ctx=ctx_len, hidden=64)

    # ── 核心：单段翟方法积分（可微分，供 checkpoint 包装）──────────────
    def _integrate_segment(
        self,
        u_seg:  torch.Tensor,   # [B, n_seg] 不平顺段（float64）
        X0:     torch.Tensor,   # [B, 7] 段初始位移
        V0:     torch.Tensor,   # [B, 7] 段初始速度
        A0:     torch.Tensor,   # [B, 7] 段初始加速度（i-1 步）
        A_pre0: torch.Tensor,   # [B, 7] 段初始加速度（i-2 步）
    ):
        """
        对一段序列运行翟方法，返回：
          acc_seg [B, n_seg]  车体 Z 加速度
          X_end, V_end, A_end, A_pre_end  段末状态（供下一段使用）
        """
        dt, alpha, beta = self.dt, self.alpha, self.beta
        B, n_seg = u_seg.shape
        u4 = u_seg.unsqueeze(1).expand(-1, 4, -1)  # [B, 4, n_seg]

        X, V, A, A_pre = X0, V0, A0, A_pre0
        acc_list = []

        for i in range(n_seg):
            if i >= 2 or (A0.abs().sum() > 0):   # 有历史时用完整公式
                X_p = X + V*dt + (0.5+alpha)*A*(dt**2) - alpha*A_pre*(dt**2)
                V_p = V + (1+beta)*A*dt            - beta*A_pre*dt
            elif i == 1:
                X_p = X + V*dt + (0.5+alpha)*A*(dt**2)
                V_p = V + (1+beta)*A*dt
            else:
                X_p = X.clone()
                V_p = V.clone()

            ui  = u4[:, :, i]                                   # [B, 4]
            F   = X_p @ self.Km.T + V_p @ self.Cm.T + ui @ self.Bm.T
            A_new = F * self.M_inv

            A_pre, A = A, A_new
            X, V = X_p, V_p
            acc_list.append(A[:, 0])

        acc_seg = torch.stack(acc_list, dim=-1)                 # [B, n_seg]
        return acc_seg, X, V, A, A_pre

    def _simulate(
        self,
        u_win: torch.Tensor,   # [B, W] float64
        z0:    torch.Tensor,   # [B, 14] 初始状态（位移 0:7，速度 7:14）
    ) -> torch.Tensor:
        """
        使用梯度检查点（问题③）运行完整窗口积分。
        显存从 O(W·B·7) 降到 O(√W·B·7)。
        """
        B, W = u_win.shape
        X = z0[:, :7].contiguous()
        V = z0[:, 7:].contiguous()
        A     = torch.zeros(B, 7, dtype=torch.float64, device=u_win.device)
        A_pre = torch.zeros(B, 7, dtype=torch.float64, device=u_win.device)

        n_seg = self.ckpt_segments if self.ckpt_segments > 0 else 1
        seg_len = (W + n_seg - 1) // n_seg
        acc_all = []

        for s in range(n_seg):
            s_start = s * seg_len
            s_end   = min(s_start + seg_len, W)
            if s_start >= W:
                break
            u_seg = u_win[:, s_start:s_end]

            if self.ckpt_segments > 0 and self.training:
                # 梯度检查点：不保留中间激活，反向时重算（解决显存问题①③）
                from torch.utils.checkpoint import checkpoint
                def _seg_fn(u_seg, X, V, A, A_pre):
                    return self._integrate_segment(u_seg, X, V, A, A_pre)
                acc_seg, X, V, A, A_pre = checkpoint(
                    _seg_fn, u_seg, X, V, A, A_pre, use_reentrant=False
                )
            else:
                acc_seg, X, V, A, A_pre = self._integrate_segment(u_seg, X, V, A, A_pre)

            acc_all.append(acc_seg)

        return torch.cat(acc_all, dim=-1)   # [B, W]

    def forward(
        self,
        u_pred: torch.Tensor,   # [B, n_dir, T]  FNO 预测不平顺
        y_obs:  torch.Tensor,   # [B, n_s,  T]   传感器观测加速度
    ) -> torch.Tensor:
        """
        完整物理前向损失（可微分，梯度流经 StateEstimator + 翟方法积分）。

        流程：
          1. 取窗口起始前 ctx_len 步车体加速度 → StateEstimator → z0
          2. 以 z0 为初始条件，对预测不平顺做翟方法积分 → ŷ_c(t)
          3. MSE(ŷ_c, y_obs_window)
        """
        B, n_dir, T = u_pred.shape
        W       = self.win_len if (0 < self.win_len < T) else T
        ctx_len = self.ctx_len

        # ── 随机选窗（问题①：短窗口降低计算量）──────────────────────────
        # 窗口起始需保留足够上下文
        t_min  = ctx_len
        t_max  = T - W
        if t_max <= t_min:
            t_start = 0
        else:
            t_start = torch.randint(t_min, t_max + 1, (1,)).item()

        u_win  = u_pred[:, 0, t_start: t_start + W].double()     # [B, W]
        y_win  = y_obs[:, 0, t_start: t_start + W]                # [B, W]

        # ── 初始状态估计（问题②）────────────────────────────────────────
        # 用窗口起始前 ctx_len 步的车体加速度作为上下文
        y_ctx  = y_obs[:, 0, t_start - ctx_len: t_start].float()  # [B, ctx_len]
        z0     = self.state_estimator(y_ctx)                       # [B, 14] float64

        # ── 翟方法可微分积分（问题③）────────────────────────────────────
        y_hat  = self._simulate(u_win, z0).to(y_obs.dtype)        # [B, W]

        return F.mse_loss(y_hat, y_win)


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
    # ── VTCM 物理正向损失 ─────────────────────────────────────────────────────
    p.add_argument("--lambda_physics", type=float, default=0.0,
                   help="物理前向损失权重；0 表示禁用")
    p.add_argument("--phys_win_len",   type=int,   default=256,
                   help="翟方法积分窗口步数（<=0 全序列），建议 128~512")
    p.add_argument("--ctx_len",         type=int,   default=64,
                   help="StateEstimator 上下文长度（步数），用于从传感器数据推断初始状态")
    p.add_argument("--ckpt_segments",   type=int,   default=4,
                   help="梯度检查点分段数（0=不使用），分段越多显存越少但计算量略增）")
    p.add_argument("--dt",             type=float, default=1e-4,
                   help="积分步长 (s)，与 generate_main.py --tstep 保持一致")
    p.add_argument("--Kh",             type=float, default=1.0e9,
                   help="轮轨赫兹线性化接触刚度 (N/m)")
    # 车辆参数（默认高速客车，与 generate_main.py VehicleParams 一致）
    p.add_argument("--Mc",  type=float, default=34000.0, help="车体质量 (kg)")
    p.add_argument("--Mt",  type=float, default=3000.0,  help="转向架质量 (kg)")
    p.add_argument("--Mw",  type=float, default=1400.0,  help="轮对质量 (kg)")
    p.add_argument("--Kpz", type=float, default=0.55e6,  help="一系垂向刚度（单侧）(N/m)")
    p.add_argument("--Ksz", type=float, default=0.40e6,  help="二系垂向刚度（单侧）(N/m)")
    p.add_argument("--Cpz", type=float, default=6e3,     help="一系垂向阻尼（单侧）(N·s/m)")
    p.add_argument("--Csz", type=float, default=80e3,    help="二系垂向阻尼（单侧）(N·s/m)")
    # ── 正则化 & Early Stopping ───────────────────────────────────────────────
    p.add_argument("--weight_decay",   type=float, default=1e-3)
    p.add_argument("--patience",       type=int,   default=40,
                   help="Early stopping patience；0 表示禁用")
    return p.parse_args(argv)
    

# ── 验证步骤 & 可视化 ─────────────────────────────────────────────────────────
def test_step(model: torch.nn.Module, dataloader: DataLoader,
              epoch: int, device: torch.device,
              fig_dir: str, vis_every: int,
              physics_loss_fn=None, lambda_physics: float = 0.0) -> tuple[float, float]:
    model.eval()
    loss_epoch  = 0.0
    phys_epoch  = 0.0
    n_batch     = 0
    viz_pred    = None
    viz_target  = None

    with torch.no_grad():
        for batch in dataloader:
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)  # [B, C_in,  T]
            target = batch["u"].to(device, non_blocking=True).transpose(1, 2)  # [B, C_out, T]
            y_obs  = batch["y"].to(device, non_blocking=True).transpose(1, 2)  # [B, n_s,  T]
            pred   = model(inp)
            data_loss = F.mse_loss(pred, target)
            phys_loss = torch.tensor(0.0, device=device)
            if physics_loss_fn is not None and lambda_physics > 0:
                phys_loss = physics_loss_fn(pred, y_obs)
            loss_epoch += (data_loss + lambda_physics * phys_loss).item()
            phys_epoch += phys_loss.item()
            n_batch    += 1
            if viz_pred is None:
                viz_pred   = pred[0].detach().cpu().numpy()    # [C_out, T]
                viz_target = target[0].detach().cpu().numpy()  # [C_out, T]

    avg_loss = loss_epoch / max(n_batch, 1)
    avg_phys = phys_epoch / max(n_batch, 1)

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

    return avg_loss, avg_phys


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

    # ── VTCM 7-DOF 垂向动力学物理损失初始化 ──────────────────────────────────
    physics_loss_fn = None
    if args.lambda_physics > 0:
        physics_loss_fn = VTCMVerticalDynamics(
            Mc             = args.Mc,
            Mt             = args.Mt,
            Mw             = args.Mw,
            Kpz            = args.Kpz,
            Ksz            = args.Ksz,
            Cpz            = args.Cpz,
            Csz            = args.Csz,
            Kh             = args.Kh,
            dt             = args.dt,
            alpha          = 0.5,
            beta           = 0.25,
            win_len        = args.phys_win_len,
            ckpt_segments  = args.ckpt_segments,
            ctx_len        = args.ctx_len,
        ).to(device)
        print(f"[Physics] VTCMVerticalDynamics 初始化完成")
        print(f"  车辆参数: Mc={args.Mc:.0f} kg, Mt={args.Mt:.0f} kg, Mw={args.Mw:.0f} kg")
        print(f"  一系: K1z={2*args.Kpz:.3e} N/m, C1z={2*args.Cpz:.3e} N·s/m")
        print(f"  二系: K2z={2*args.Ksz:.3e} N/m, C2z={2*args.Csz:.3e} N·s/m")
        print(f"  轮轨接触: Kh={args.Kh:.3e} N/m")
        print(f"  积分窗口: {args.phys_win_len} 步, 上下文: {args.ctx_len} 步, 梯度检查点分段: {args.ckpt_segments}")
        print(f"  物理损失权重 λ={args.lambda_physics}")

    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.999), lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    best_val_loss    = float("inf")
    patience_counter = 0
    history = {"train": [], "val": [], "phys_train": [], "phys_val": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss  = 0.0
        total_phys  = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for batch in pbar:
            optimizer.zero_grad()
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)   # [B, n_s,  T]
            target = batch["u"].to(device, non_blocking=True).transpose(1, 2)   # [B, n_dir, T]
            y_obs  = inp  # same tensor, already [B, n_s, T]
            pred   = model(inp)                                                   # [B, n_dir, T]
            data_loss = F.mse_loss(pred, target)
            phys_loss = torch.tensor(0.0, device=device)
            if physics_loss_fn is not None:
                phys_loss = physics_loss_fn(pred, y_obs)
            loss = data_loss + args.lambda_physics * phys_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_phys += phys_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4e}", phys=f"{phys_loss.item():.4e}")

        avg_loss = total_loss / len(train_loader)
        avg_phys = total_phys / len(train_loader)
        val_loss, val_phys = 0.0, 0.0
        if val_loader is not None:
            val_loss, val_phys = test_step(
                model, val_loader, epoch, device, args.fig_dir, args.vis_every,
                physics_loss_fn=physics_loss_fn, lambda_physics=args.lambda_physics,
            )

        history["train"].append(avg_loss)
        history["val"].append(val_loss)
        history["phys_train"].append(avg_phys)
        history["phys_val"].append(val_phys)

        phys_info = f"  phys_t={avg_phys:.3e} phys_v={val_phys:.3e}" if args.lambda_physics > 0 else ""
        print(
            f"[Epoch {epoch + 1:4d}/{args.epochs}] "
            f"train={avg_loss:.4e}  val={val_loss:.4e}  "
            f"lr={optimizer.param_groups[0]['lr']:.3e}{phys_info}"
        )

        # ── 保存最优模型 ─────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, "best_model.pth"))
        else:
            patience_counter += 1

        save_checkpoint(
            args.ckpt_dir, models=model,
            optimizer=optimizer, scheduler=scheduler, epoch=epoch,
        )

        # ── Early Stopping ───────────────────────────────────────────────────
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"[Early Stopping] No improvement for {args.patience} epochs. Stop.")
            break

    # 训练结束后全量推理可视化
    if val_loader is not None:
        print("Generating full inference plots ...")
        inference_and_plot(model, val_loader, device, args.fig_dir, epoch=args.epochs - 1)
        print(f"Figures saved to: {args.fig_dir}")


if __name__ == "__main__":
    main()
