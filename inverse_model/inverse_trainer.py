'''
Author: 2099193635 2099193635@qq.com
Date: 2026-04-07 16:17:56
LastEditors: 2099193635 2099193635@qq.com
LastEditTime: 2026-04-12 16:34:14
FilePath: /VTCM_PYTHON/inverse_model/inverse_trainer.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import torch
from typing import Optional
from differentiable_physics import PhysicsLayer
from torch import Tensor
from inverse_architecture import InverseOperator
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import math
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)




def track_psd_standard(freqs_spatial: Tensor, grade: int = 2) -> Tensor:
    """
    Standard PSD of track irregularity for a given track grade.

    PSD(kappa) ≈ A_v · (kappa_0 / kappa)^w  [m^3/cycle]

    Args:
        freqs_spatial: [F], the spatial frequencies at which to evaluate the PSD
        grade: track grade (1, 2, 3, 4)
    
    Returns:
        psd_ref: [F], the standard PSD values corresponding to the input frequencies
    """
    A_v = {1: 3.363e-7, 2: 1.080e-6, 3: 4.032e-6, 4: 8.532e-6}.get(grade, 1.080e-6)
    kappa_0 = 0.0206
    w = 2.0
    psd = A_v * (kappa_0 / freqs_spatial.clamp(min=1e-6)) ** w
    return psd.clamp(min=1e-30)


def compute_psd(z: Tensor, dx: float) -> tuple[Tensor, Tensor]:
    """
    Oneside Power Spectral Density (PSD) esitimation.

    Args:
        z: [B, L], the track irregularity data
        dx: spatial resolution of the track irregularity data
    
    Returns:
        freqs: [F]  the frequencies corresponding to the PSD values
        psd:   [B, F] or [B, F, n_dir] the estimated PSD values for each sample in the batch
    """
    if z.dim() == 2:
        z = z.unsqueeze(dim=-1)  # [B, L, 1]
    B, L, D = z.shape

    z_f = torch.fft.rfft(z, dim=1)          # [B, F, D], F = L//2 + 1
    psd = (z_f.abs() ** 2) * (2 * dx / L)   # [B, F, D], oneside PSD estimation
    psd[:, 0, :] /= 2                       # DC component should not be doubled
    if L % 2 == 0:
        psd[:, -1, :] /= 2                  # Nyquist component should not be doubled for even L
    freqs = torch.fft.rfftfreq(L, d=dx, device = z.device)    # [F]
    return freqs, psd.squeeze(dim=-1) if D == 1 else psd


def spectral_kl_loss(
        z_pred: Tensor,  # [B, L]
        grade: int = 2,
        dx: float = 1
)-> Tensor:
    """
    KL(PSD_pred || PSD_standard)  Spectral Prior Loss (Log-Domain Symmetric Approximation)
    
    Args:
        z_pred:  [B, L] or [B, L, n_dir], the predicted track irregularity from the Inverse model
        grade:   track grade
        dx:      spatial resolution of the track irregularity data
    Returns:
        loss:    scale
    """
    freqs, psd_pred = compute_psd(z_pred, dx)  # [B, n_freq]
    psd_ref = track_psd_standard(freqs, grade=grade)  # [n_freq]

    psd_pred_nz = psd_pred[:, 1:].clamp(min=1e-30)  # avoid log(0), ignore DC component
    psd_ref_nz = psd_ref[1:].clamp(min=1e-30)        # [n_freq-1]
    
    log_ratio = torch.log(psd_pred_nz) - torch.log(psd_ref_nz.unsqueeze(0))  # [B, n_freq-1]
    return (log_ratio ** 2).mean()  # MSE in log-domain


def plot_predictions(
    u_pred: Tensor,        # [B, L] or [B, L, n_dir]
    u_true: Tensor,        # [B, L] or [B, L, n_dir]
    epoch: int,
    save_dir: str | Path,
    seq_lengths: Optional[Tensor] = None,
    n_samples: int = 4,
) -> None:
    """
    绘制预测值与真实值的对比曲线图。
    
    Args:
        u_pred: 预测的不平顺 [B, L] 或 [B, L, n_dir]
        u_true: 真实的不平顺 [B, L] 或 [B, L, n_dir]
        epoch: 训练轮数
        save_dir: 保存图像的目录
        n_samples: 绘制的样本数量（取前n_samples个）
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 移到CPU并转为numpy
    u_pred = u_pred.detach().cpu().numpy()  # [B, L] or [B, L, n_dir]
    u_true = u_true.detach().cpu().numpy()  # [B, L] or [B, L, n_dir]
    seq_lengths_np = None
    if seq_lengths is not None:
        seq_lengths_np = seq_lengths.detach().cpu().numpy()
    
    B = u_pred.shape[0]
    n_samples = min(n_samples, B)
    
    # 如果有多个方向，取垂直方向(第一列)
    if u_pred.ndim == 3:
        u_pred = u_pred[:, :, 0]  # [B, L]
        u_true = u_true[:, :, 0]  # [B, L]
    
    # 创建子图
    ncols = 2
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        current_len = int(seq_lengths_np[i]) if seq_lengths_np is not None else u_pred.shape[1]
        x_axis = np.arange(current_len)
        pred_i = u_pred[i, :current_len]
        true_i = u_true[i, :current_len]
        
        ax.plot(x_axis, true_i, 'b-', label='Ground Truth', linewidth=1.5, alpha=0.8)
        ax.plot(x_axis, pred_i, 'r--', label='Prediction', linewidth=1.5, alpha=0.8)
        
        mse = np.mean((pred_i - true_i) ** 2)
        rmse = np.sqrt(mse)
        
        ax.set_title(f"Sample {i+1}  |  RMSE: {rmse:.4e}", fontsize=10)
        ax.set_xlabel("Spatial Index")
        ax.set_ylabel("Irregularity (m)")
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_samples, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f"Epoch {epoch}: Predicted vs Ground Truth Irregularity", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"pred_vs_true_epoch_{epoch:04d}.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction plot saved: {save_path}")


class InverseTrainer:
    """
    The main loop of InverseOperator training 
    Args:
        model: InverseOperator model to be trained
        cfg:   training configuration
        forward_model: Physics infromed part, which is used to compute the ODE results: [B, L, n_dir]→[B, T, n_s], whcih is optional
        device: CPU or GPU
    """

    def __init__(self, 
                 model: InverseOperator, 
                 cfg, 
                 forward_model: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 )-> None:
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        mode = getattr(cfg, "physics_mode", "frf")
        
        # ======= Differentiable Physics Layer (frf / pinn / both)========
        self.physics_layer: Optional[PhysicsLayer] = None
        if mode in ("frf", "pinn", "both"):
            self.physics_layer = PhysicsLayer(
                mode=mode,
                n_modal=cfg.n_modal,
                n_sensors=cfg.n_sensors,
                dx=getattr(cfg, "dx", 0.25),
                dt=getattr(cfg, "dt", 0.005),
                EI=getattr(cfg, "track_EI", 6.4e6),
                k_b=getattr(cfg, "track_k_b", 5.0e7),
                k_Hz=getattr(cfg, "track_k_Hz", 1.2e9),
                frf_learnable=getattr(cfg, "frf_learnable", True),
            ).to(self.device)
            logger.info(f"Differentiable physics layer initialized (mode={mode}).")
        
        # Optimizer (including learnable parameters of the physics_layer, such as modal residues Ri)
        params = list(self.model.parameters())
        if self.physics_layer is not None:
            params += list(self.physics_layer.parameters())
            
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=cfg.lr_gamma
        )


    def _compute_loss(self,
                      z_pred: Tensor,  # [B, L, n_dir]
                      z_true: Tensor,  # [B, L, n_dir]
                      a_c: Tensor,     # [B, T, n_s]
                      seq_lengths: Optional[Tensor],
                      cond: Tensor,    # [B, n_cond] 
                      ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute three losses.

        Args:
            z_pred: [B, L, n_dir], the predicted track irregularity from the Inverse model
            z_true: [B, L, n_dir], the true track irregularity from the dataset
            a_c: [B, T, n_s], the observed trajectory of the system, which is used for physics-informed loss
            cond:
                cond[:, :n_modal]            =  Natural Circular Frequency omega_i [rad/s]
                cond[:, n_modal : 2*n_modal] =  Damping Ratio zeta_i
                cond[:, 2*n_modal]           =  velocity v_x [m/s]
        """

        mode = getattr(self.cfg, "physics_mode", "frf")
        n_modal = self.cfg.n_modal
        nat_freqs = cond[:, :n_modal]                 # [B, n_modal]
        damp_ratios = cond[:, n_modal : 2*n_modal]    # [B, n_modal]
        vx_mps = cond[:, 2 * n_modal]                 # [B]

        # ========== L_data (MSE) =========
        if seq_lengths is not None:
            max_len = z_pred.size(1)
            valid_mask = (
                torch.arange(max_len, device=z_pred.device).unsqueeze(0)
                < seq_lengths.to(z_pred.device).unsqueeze(1)
            ).unsqueeze(-1).to(z_pred.dtype)
            diff2 = ((z_pred - z_true) ** 2) * valid_mask
            denom = valid_mask.sum().clamp(min=1.0)
            l_data = diff2.sum() / denom
        else:
            l_data = F.mse_loss(z_pred, z_true)

        # ========== L_phys (physics-informed loss) =========
        l_phys = torch.tensor(0.0, device=self.device)
        l_frf = torch.tensor(0.0, device=self.device)
        l_pinn = torch.tensor(0.0, device=self.device)

        if mode in ("frf", "pinn", "both") and self.physics_layer is not None:
            z_vert = z_pred[:, :, 0] if z_pred.dim() == 3 else z_pred # [B, L, 1]
            l_phys, phys_m = self.physics_layer(
                z_pred=z_vert,
                a_c_true=a_c,
                nat_freqs=nat_freqs,
                damp_ratios=damp_ratios,
                vx_mps=vx_mps,
                lambda_frf=getattr(self.cfg, "lambda_frf", 1.0),
                lambda_pinn=getattr(self.cfg, "lambda_pinn", 1.0),
            )
            l_frf = torch.tensor(phys_m["l_frf"], device=self.device)
            l_pinn = torch.tensor(phys_m["l_pinn"], device=self.device)
        
        # ========== L_spec (KL PSD) =========
        l_spec = torch.tensor(0.0, device=self.device)
        if self.cfg.lambda_spec > 0:
            z_vert = z_pred[:, :, 0] if z_pred.dim() == 3 else z_pred # [B, L, 1]
            if seq_lengths is not None:
                spec_terms = []
                seq_lengths_list = seq_lengths.detach().cpu().tolist()
                for z_i, seq_len in zip(z_vert, seq_lengths_list):
                    spec_terms.append(
                        spectral_kl_loss(
                            z_pred=z_i[:int(seq_len)].unsqueeze(0),
                            grade=getattr(self.cfg, "track_grade", 2),
                            dx=getattr(self.cfg, "dx", 0.25),
                        )
                    )
                l_spec = torch.stack(spec_terms).mean() if spec_terms else l_spec
            else:
                l_spec = spectral_kl_loss(
                    z_pred=z_vert,
                    grade=getattr(self.cfg, "track_grade", 2),
                    dx=getattr(self.cfg, "dx", 0.25),
                )

        # ========== L_spatial (空间变化约束，核心) =========
        # 防止输出退化为常数：强制预测值具有足够的空间变化
        # 这是最关键的约束，因为数据损失本身会让模型学到均值预测
        l_spatial = torch.tensor(0.0, device=self.device)
        lambda_spatial = getattr(self.cfg, "lambda_spatial", 1.0)  # 提高默认权重到1.0
        if lambda_spatial > 0:
            z_vert_pred = z_pred[:, :, 0] if z_pred.dim() == 3 else z_pred  # [B, L]
            z_vert_true = z_true[:, :, 0] if z_true.dim() == 3 else z_true
            
            if seq_lengths is not None:
                # 按seq_lengths计算每个样本的有效空间std
                spatial_terms = []
                seq_lengths_list = seq_lengths.detach().cpu().tolist()
                for pred_i, true_i, seq_len in zip(z_vert_pred, z_vert_true, seq_lengths_list):
                    seq_len = int(seq_len)
                    std_pred = pred_i[:seq_len].std()
                    std_true = true_i[:seq_len].std()
                    # 目标：std_pred >= 0.8 * std_true（要求80%的目标变化）
                    # 如果不满足，返回巨大的惩罚（指数增长）
                    gap = 0.8 * std_true - std_pred
                    if gap > 0:
                        # 未达到目标时，惩罚为gap的平方，强制优化
                        spatial_terms.append(gap ** 2)
                    else:
                        # 超过目标时，无惩罚
                        spatial_terms.append(torch.tensor(0.0, device=self.device))
                l_spatial = torch.stack(spatial_terms).mean() if spatial_terms else l_spatial
            else:
                std_pred = z_vert_pred.std(dim=1)
                std_true = z_vert_true.std(dim=1)
                gap = 0.8 * std_true - std_pred
                l_spatial = torch.nn.functional.relu(gap).mean() ** 2

        # ========== Weighted Fusion =========
        lambda_phys = 1
        loss_total = l_data + lambda_phys * l_phys + self.cfg.lambda_spec * l_spec + lambda_spatial * l_spatial

        return loss_total, {
            "l_total": loss_total.item(),
            "l_data": l_data.item(),
            "l_phys": l_phys.item(),
            "l_frf": l_frf.item(),
            "l_pinn": l_pinn.item(),
            "l_spec": l_spec.item(),
            "l_spatial": l_spatial.item(),
        }


    # ================= One epoch of training =================
    def _run_epoch(
            self, loader: DataLoader, train: bool = True    
    )-> dict[str, float]:
        """
        Run one epoch of training or evaluation.
        
        Args:
            loader: DataLoader for the training or evaluation dataset
            train: whether to run in training mode (with backpropagation) or evaluation mode (without backpropagation)
        Returns:
            A dictionary of averaged metrics for the epoch, including:
                - "loss_total": the total loss (data + physics + spectral)
                - "l_data": the data loss (MSE)
                - "l_phys": the physics-informed loss
                - "l_frf": the FRF loss component (if applicable)
                - "l_pinn": the PINN loss component (if applicable)
                - "l_spec": the spectral prior loss (if applicable)
        """
        self.model.train(train)
        if self.physics_layer is not None:
            self.physics_layer.train(train)
        
        keys = ["l_data", "l_phys", "l_frf", "l_pinn", "l_spec", "l_spatial", "l_total"]
        agg: dict[str, float] = {k: 0.0 for k in keys}
        n = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                y = batch["y"].to(self.device)          # [B, T, n_s]
                u = batch["u"].to(self.device)          # [B, L, n_dir]
                c = batch["c"].to(self.device)          # [B, n_cond]
                xq = batch["x_query"].to(self.device)   # [B, L]
                seq_lengths = batch.get("seq_lengths")
                if seq_lengths is not None:
                    seq_lengths = seq_lengths.to(self.device)

                z_pred = self.model(y, xq, c)           # [B, L, n_dir]
                loss, metrics = self._compute_loss(z_pred, u, y, seq_lengths, c)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient Clipping
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters()]
                        + (list(self.physics_layer.parameters()) if self.physics_layer else []),
                        max_norm=1.0,
                    )
                    self.optimizer.step()
                
                B = y.size(0)
                for k, v in metrics.items():
                    agg[k] += v * B
                n += B
        return {k: v / n for k, v in agg.items()}
    
    # ================= Main training loop =================

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        ckpt_dir: Optional[str | Path] = None,
        log_every: int = 10,
        val_every: int = 5,
    ) -> dict[str, list[float]]:
        """
        Main training loop.

        Args:
            train_loader:   DataLoader for the training dataset
            val_loader:     DataLoader for the validation dataset (optional)
            epochs:         number of epochs to train
            ckpt_dir:       directory to save model checkpoints (optional)
            log_every:      frequency of logging metrics (in epochs)
            val_every:      frequency of validation and visualization (in epochs)
        
        Returns:
            history: a dictionary containing the history of metrics during training, including:
                - "train_loss_total": list of total loss values on the training set for each epoch
                - "train_l_data": list of data loss values on the training set for each epoch
                - "train_l_phys": list of physics-informed loss values on the training set for each epoch
                - "train_l_spec": list of spectral prior loss values on the training set for each epoch
                - "val_loss_total": list of total loss values on the validation set for each epoch (if val_loader is provided)
                - "val_l_data": list of data loss values on the validation set for each epoch (if val_loader is provided)
                - "val_l_phys": list of physics-informed loss values on the validation set for each epoch (if val_loader is provided)
                - "val_l_spec": list of spectral prior loss values on the validation set for each epoch (if val_loader is provided)
        """

        if ckpt_dir is not None:
            ckpt_dir = Path(ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            vis_dir = ckpt_dir / "predictions"
            vis_dir.mkdir(exist_ok=True)
        else:
            vis_dir = None

        best_val = float("inf")
        history: dict[str, list[float]] = {}

        for epoch in range(1, epochs + 1):
            train_m = self._run_epoch(train_loader, train=True)
            self.scheduler.step()

            for k, v in train_m.items():
                history.setdefault(f"train_{k}", []).append(v)
            
            # ──── Validation & Visualization every val_every epochs ────
            if epoch % val_every == 0 and val_loader is not None:
                val_m = self._run_epoch(val_loader, train=False)
                for k, v in val_m.items():
                    history.setdefault(f"val_{k}", []).append(v)

                if epoch % log_every == 0:
                    logger.info(
                        f"Epoch {epoch:4d} | "
                        f"train {train_m['l_total']:.4f} "
                        f"(data {train_m['l_data']:.4f} "
                        f"frf {train_m['l_frf']:.4f} "
                        f"pinn {train_m['l_pinn']:.4f} "
                        f"spec {train_m['l_spec']:.4f}) | "
                        f"val {val_m['l_total']:.4f}"
                    )
                
                # ──── Generate prediction visualization ────
                if vis_dir is not None:
                    self._visualize_predictions(val_loader, epoch, vis_dir)

            
            if ckpt_dir is not None and epoch % 20 == 0:
                self._save_checkpoint(
                    ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt", epoch
                )

        if ckpt_dir is not None:
            self._save_checkpoint(ckpt_dir / "checkpoint_latest.pt", epochs)
        return history

    def _visualize_predictions(
        self,
        val_loader: DataLoader,
        epoch: int,
        vis_dir: str | Path,
    ) -> None:
        """
        在验证集上进行前向推理并绘制预测值与真实值的对比曲线。
        
        Args:
            val_loader: 验证集DataLoader
            epoch: 当前轮数
            vis_dir: 保存图像的目录
        """
        self.model.eval()
        
        u_preds_all = []
        u_trues_all = []
        seq_lengths_all = []
        
        with torch.no_grad():
            for batch in val_loader:
                y = batch["y"].to(self.device)          # [B, T, n_s]
                u = batch["u"].to(self.device)          # [B, L, n_dir]
                c = batch["c"].to(self.device)          # [B, n_cond]
                xq = batch["x_query"].to(self.device)   # [B, L]
                seq_lengths = batch.get("seq_lengths")
                
                z_pred = self.model(y, xq, c)           # [B, L, n_dir]
                
                u_preds_all.append(z_pred.detach().cpu())
                u_trues_all.append(u.detach().cpu())
                if seq_lengths is not None:
                    seq_lengths_all.append(seq_lengths.detach().cpu())
        
        # 拼接所有样本
        u_pred_all = torch.cat(u_preds_all, dim=0)  # [N, L, n_dir]
        u_true_all = torch.cat(u_trues_all, dim=0)  # [N, L, n_dir]
        seq_lengths_all_tensor = torch.cat(seq_lengths_all, dim=0) if seq_lengths_all else None
        
        # 绘制对比图
        plot_predictions(
            u_pred=u_pred_all,
            u_true=u_true_all,
            epoch=epoch,
            save_dir=vis_dir,
            seq_lengths=seq_lengths_all_tensor,
            n_samples=4,
        )

    def _save_checkpoint(
            self, path: Path, epoch: int, best_metric: Optional[float] = None
    ) -> None:

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": best_metric,
            "config": self.cfg,
        }
        if self.physics_layer is not None:
            state["physics_layer_state_dict"] = self.physics_layer.state_dict()
        torch.save(state, path)
        logger.debug(f"Checkpoint saved: {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        model: InverseOperator,
        cfg,
        forward_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> "InverseTrainer":
        """Load a checkpoint and resume training."""
        ckpt = torch.load(path, map_location="cpu")
        trainer = cls(model, cfg, forward_model=forward_model, device=device)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if trainer.physics_layer is not None and "physics_layer_state_dict" in ckpt:
            trainer.physics_layer.load_state_dict(ckpt["physics_layer_state_dict"])
        logger.info(f"Resumed from epoch {ckpt['epoch']} (path: {path})")
        return trainer