'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-03-24 06:12:12
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-03-26 02:59:11
FilePath: /VTCM_PYTHON/PINO/VTCM_PINO.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''

import sys
import os
import time
from pathlib import Path

# 添加 physicsnemo 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "physicsnemo"))

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.utils.checkpoint import save_checkpoint
from physicsnemo.models.fno import FNO
from WQJ.models.fourier1d import FNN1d_VTCD_GradNorm_Branch
from omegaconf import DictConfig
from torch.utils.data import DataLoader   
from tqdm.auto import tqdm
from pino_utils import VTCMHDF5MapStyleDataset

# 目标输出通道（从原21通道中抽取）:
# [carbody_z, bogie_front_z, bogie_rear_z] x [disp, vel, acc]
SELECTED_CHANNEL_INDICES = [0, 1, 2, 7, 8, 9, 14, 15, 16]
POSITION_LABELS = ["Carbody (Z)", "Front Bogie (Z)", "Rear Bogie (Z)"]
COMPONENT_LABELS = ["Displacement", "Velocity", "Acceleration"]


class FNOBranchVTCD(torch.nn.Module):
    """完整复刻 Runner 结构（FNN1d_VTCD_GradNorm_Branch）并适配当前数据维度。"""
    
    def __init__(
        self,
        in_channels: int,
        out_dim1: int,
        out_dim2: int,
        cfg: DictConfig,
        task_number: int = 3,
    ) -> None:
        super().__init__()
        
        # 直接使用 Runner 的原生网络结构参数
        m = cfg.model
        self.runner_branch = FNN1d_VTCD_GradNorm_Branch(
            modes1=int(m.modes1),
            modes2=int(m.modes2),
            width1=int(m.width1),
            width2=int(m.width2),
            fc_dim1=int(m.fc_dim1),
            fc_dim2=int(m.fc_dim2),
            inputDim1=int(m.inputDim1),
            inputDim2=int(m.inputDim2),
            outputDim1=int(out_dim1),
            outputDim2=int(out_dim2),
            task_number=task_number,
        )

        # 暴露同名属性，保持后续 GradNorm 代码兼容
        self.task_weights = self.runner_branch.task_weights

    def forward(self, x: torch.Tensor):
        # 当前数据: x=[B,C,T]；Runner网络期望: [B,T,C]
        x_bt_c = x.permute(0, 2, 1).contiguous()

        out1_bt_c, out2_bt_c = self.runner_branch(x_bt_c)  # each [B,T,Ci]

        # 转回当前训练流程使用的 [B,C,T]
        out1 = out1_bt_c.permute(0, 2, 1).contiguous()
        out2 = out2_bt_c.permute(0, 2, 1).contiguous()
        out = torch.cat([out1, out2], dim=1)
        return out, out1, out2


def _split_dva(y: torch.Tensor):
    """Split [B, 9, T] -> disp/vel/acc each [B, 3, T]."""
    disp = y[:, 0:3, :]
    vel = y[:, 3:6, :]
    acc = y[:, 6:9, :]
    return disp, vel, acc


def _first_order_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Forward difference along time dim, output shape [B, C, T-1]."""
    return (x[:, :, 1:] - x[:, :, :-1]) / dt


def compute_physics_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    dt: float,
) -> dict:
    """
    Physics-informed losses (Runner-like):
      1) Kinematic consistency: d/dt(disp_pred)≈vel_pred, d/dt(vel_pred)≈acc_pred
      2) Derivative supervision: d/dt(pred)≈d/dt(target), d2/dt2(pred)≈d2/dt2(target)
    """
    p_disp, p_vel, p_acc = _split_dva(pred)
    t_disp, t_vel, t_acc = _split_dva(target)

    # ---- kinematic consistency from prediction itself ----
    p_vel_fd = _first_order_diff(p_disp, dt)          # [B,3,T-1]
    p_acc_fd = _first_order_diff(p_vel, dt)           # [B,3,T-1]
    kin_disp_vel = F.mse_loss(p_vel[:, :, 1:], p_vel_fd)
    kin_vel_acc = F.mse_loss(p_acc[:, :, 1:], p_acc_fd)
    loss_kinematic = kin_disp_vel + kin_vel_acc

    # ---- derivative supervision (similar to Runner differential losses) ----
    p_disp_d1 = _first_order_diff(p_disp, dt)
    t_disp_d1 = _first_order_diff(t_disp, dt)
    p_vel_d1 = _first_order_diff(p_vel, dt)
    t_vel_d1 = _first_order_diff(t_vel, dt)

    loss_diff_1st = F.mse_loss(p_disp_d1, t_disp_d1) + F.mse_loss(p_vel_d1, t_vel_d1)

    p_disp_d2 = _first_order_diff(p_disp_d1, dt)
    t_disp_d2 = _first_order_diff(t_disp_d1, dt)
    loss_diff_2nd = F.mse_loss(p_disp_d2, t_disp_d2)

    return {
        "loss_kinematic": loss_kinematic,
        "loss_diff_1st": loss_diff_1st,
        "loss_diff_2nd": loss_diff_2nd,
    }


def _resolve_h5_path(path_str: str) -> str:
    """Resolve HDF5 path robustly under Hydra chdir=True."""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    # 项目根目录：/workspace/VTCM_PYTHON
    project_root = Path(__file__).resolve().parents[1]
    return str((project_root / p).resolve())

def test_step(model, dataloader, epoch, device, selected_channel_indices):
    model.eval()
    loss_epoch = 0.0
    n_batch = 0
    t_eval_start = time.perf_counter()

    viz_pred = None
    viz_target = None

    with torch.no_grad():
        for batch in dataloader:
            input = batch["input"].to(device, non_blocking=True)
            output = batch["output"].to(device, non_blocking=True)
            output = output[:, selected_channel_indices, :]

            model_out = model(input)
            pred = model_out[0] if isinstance(model_out, tuple) else model_out
            loss = F.mse_loss(pred, output)

            loss_epoch += loss.item()
            n_batch += 1

            # 仅缓存首个 batch 的第一个样本用于可视化
            if viz_pred is None:
                viz_pred = pred[0].detach().cpu().numpy()      # [C, T]
                viz_target = output[0].detach().cpu().numpy()  # [C, T]

    avg_loss = loss_epoch / max(n_batch, 1)
    eval_time = time.perf_counter() - t_eval_start

    # 每 10 个 epoch 可视化一次预测效果（SCI风格折线图）
    if viz_pred is not None and (epoch % 10 == 0):
        os.makedirs("figures", exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 11,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "legend.fontsize": 9,
                "lines.linewidth": 1.4,
                "figure.dpi": 150,
            }
        )

        # local channel indices in 9-ch output:
        # disp:0..2, vel:3..5, acc:6..8
        component_offsets = [0, 3, 6]

        for pos_idx, pos_name in enumerate(POSITION_LABELS):
            fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
            t_axis = np.arange(viz_pred.shape[1])

            for comp_i, comp_name in enumerate(COMPONENT_LABELS):
                ch = component_offsets[comp_i] + pos_idx
                axes[comp_i].plot(t_axis, viz_target[ch], color="#1f77b4", label="Ground Truth")
                axes[comp_i].plot(t_axis, viz_pred[ch], color="#d62728", linestyle="--", label="Prediction")
                axes[comp_i].set_ylabel(comp_name)
                axes[comp_i].set_title(f"{pos_name} - {comp_name}")
                axes[comp_i].grid(True, alpha=0.35)
                axes[comp_i].spines["top"].set_visible(False)
                axes[comp_i].spines["right"].set_visible(False)
                if comp_i == 0:
                    axes[comp_i].legend(frameon=False, loc="upper right")

            axes[-1].set_xlabel("Time Index")
            fig.suptitle(f"Validation Comparison at Epoch {epoch + 1}: {pos_name}", y=0.995)
            fig.tight_layout()
            pos_file = pos_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            fig.savefig(f"figures/val_epoch_{epoch:04d}_{pos_file}.png", dpi=300)
            plt.close(fig)

    print(
        f"[Epoch {epoch + 1}] valid done | batches={n_batch}, "
        f"mse={avg_loss:.6e}, time={eval_time:.2f}s"
    )
    return avg_loss

@hydra.main(version_base = "1.3", config_path = "conf", config_name='vtcm_pino_runner_branch.yaml')
def main(cfg:DictConfig):
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    LaunchLogger.initialize()
    num_workers = int(getattr(cfg, "num_workers", 4))
    preload_hdf5 = bool(getattr(cfg, "preload_hdf5", True))
    pin_memory = bool(device.type == "cuda")
    train_h5 = _resolve_h5_path(str(cfg.train_hdf5))
    validation_h5 = _resolve_h5_path(str(cfg.val_hdf5))
    test_h5_cfg = str(getattr(cfg, "test_hdf5", cfg.val_hdf5))
    test_h5 = _resolve_h5_path(test_h5_cfg)
    train_dataset = VTCMHDF5MapStyleDataset(
        train_h5,
        device="cpu",
        preload_to_memory=preload_hdf5,
        return_meta=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_dataset = VTCMHDF5MapStyleDataset(
        test_h5,
        device="cpu",
        preload_to_memory=preload_hdf5,
        return_meta=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    validation_dataset = VTCMHDF5MapStyleDataset(
        validation_h5,
        device="cpu",
        preload_to_memory=preload_hdf5,
        return_meta=False,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    print("=" * 90)
    print(f"Run directory: {Path.cwd()}")
    print(f"Device: {device}")
    print(
        f"Config | batch_size={cfg.batch_size}, num_workers={num_workers}, "
        f"preload_hdf5={preload_hdf5}, pin_memory={pin_memory}"
    )
    print(
        f"Dataset | train={len(train_dataset)}, test={len(test_dataset)}, "
        f"val={len(validation_dataset)}"
    )
    print(
        f"Dataloader batches | train={len(train_dataloader)}, test={len(test_dataloader)}, "
        f"val={len(validation_dataloader)}"
    )
    print(f"HDF5 paths | train={train_h5}")
    print(f"HDF5 paths | test={test_h5}")
    print(f"HDF5 paths | val={validation_h5}")
    print(f"Selected output channels (from 21): {SELECTED_CHANNEL_INDICES}")
    print("Target outputs: Carbody/FrontBogie/RearBogie (Z) x [disp, vel, acc] => 9 channels")
    print(
        f"Physics loss config | physics_weight={float(getattr(cfg, 'physics_weight', 0.1))}, "
        f"dynamics_weight={float(cfg.loss.dynamics_weight)}, "
        f"kinematic_weight={float(cfg.loss.kinematic_weight)}, dt={float(cfg.dynamics.dt)}"
    )
    print("=" * 90)

    in_channels = train_dataset[0]["input"].shape[0]
    out_channels = len(SELECTED_CHANNEL_INDICES)
    out_dim1 = int(cfg.model.out_dim1)
    out_dim2 = int(cfg.model.out_dim2)
    if out_dim1 + out_dim2 != out_channels:
        raise ValueError(
            f"out_dim1 + out_dim2 必须等于 {out_channels}，当前为 {out_dim1}+{out_dim2}"
        )

    # Runner同构：双分支模型 + task_weights
    task_number = 3  # data / diff / kin
    model = FNOBranchVTCD(
        in_channels=in_channels,
        out_dim1=out_dim1,
        out_dim2=out_dim2,
        cfg=cfg,
        task_number=task_number,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    # Runner风格损失权重
    gradnorm_on = str(cfg.data.GradNorm).lower() == "on"
    xy_loss_w = float(cfg.train.xy_loss)
    f_loss_w = float(cfg.train.f_loss)
    diff1_w = float(cfg.train.diff_loss1)
    diff2_w = float(cfg.train.diff_loss2)
    carbody_w = float(cfg.train.carbody_loss)
    others_w = float(cfg.train.others_loss)
    w_kin = float(cfg.loss.kinematic_weight)
    w_dyn = float(cfg.loss.dynamics_weight)
    dt = float(cfg.dynamics.dt)

    # carbody为每组分量第一通道：disp/vel/acc -> [0,3,6]
    carbody_idx = [0, 3, 6]
    others_idx = [i for i in range(out_channels) if i not in carbody_idx]

    for epoch in range(cfg.max_epochs):
        t_epoch_start = time.perf_counter()
        total_loss = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0
        total_kin_loss = 0.0
        total_diff1_loss = 0.0
        total_diff2_loss = 0.0
        t_data_to_gpu = 0.0
        t_forward = 0.0
        t_backward = 0.0
        t_optim = 0.0

        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(train_dataloader),
            epoch_alert_freq=10,
        ) as log:
            pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{cfg.max_epochs}",
                total=len(train_dataloader),
                leave=False,
            )
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()

                t0 = time.perf_counter()
                input = batch["input"].to(device, non_blocking=True)
                output = batch["output"].to(device, non_blocking=True)
                output = output[:, SELECTED_CHANNEL_INDICES, :]
                t_data_to_gpu += time.perf_counter() - t0

                t1 = time.perf_counter()
                pred, pred_1, pred_2 = model(input)

                # Runner里的 carbody/others 分项数据权重
                data_loss_carbody = F.mse_loss(pred[:, carbody_idx, :], output[:, carbody_idx, :])
                data_loss_others = F.mse_loss(pred[:, others_idx, :], output[:, others_idx, :])
                data_loss = carbody_w * data_loss_carbody + others_w * data_loss_others

                phys_terms = compute_physics_losses(pred, output, dt=dt)

                # Runner风格多任务总损失：data + diff + kinematic
                task_data = xy_loss_w * data_loss
                task_diff = f_loss_w * (
                    w_dyn * (diff1_w * phys_terms["loss_diff_1st"] + diff2_w * phys_terms["loss_diff_2nd"])
                )
                task_kin = w_kin * phys_terms["loss_kinematic"]

                if gradnorm_on:
                    tw = torch.softmax(model.task_weights, dim=0)
                    loss = tw[0] * task_data + tw[1] * task_diff + tw[2] * task_kin
                else:
                    loss = task_data + task_diff + task_kin

                phys_loss = task_diff + task_kin
                t_forward += time.perf_counter() - t1

                t2 = time.perf_counter()
                loss.backward()
                t_backward += time.perf_counter() - t2

                t3 = time.perf_counter()
                optimizer.step()
                scheduler.step()
                t_optim += time.perf_counter() - t3

                total_loss += loss.item()
                total_data_loss += data_loss.item()
                total_phys_loss += phys_loss.item()
                total_kin_loss += phys_terms["loss_kinematic"].item()
                total_diff1_loss += phys_terms["loss_diff_1st"].item()
                total_diff2_loss += phys_terms["loss_diff_2nd"].item()

                if (batch_idx + 1) % 20 == 0:
                    pbar.set_postfix(
                        {
                            "total": f"{loss.item():.3e}",
                            "data": f"{data_loss.item():.3e}",
                            "phys": f"{phys_loss.item():.3e}",
                        }
                    )

            n_train_batch = len(train_dataloader)
            avg_loss = total_loss / n_train_batch
            avg_data_loss = total_data_loss / n_train_batch
            avg_phys_loss = total_phys_loss / n_train_batch
            avg_kin_loss = total_kin_loss / n_train_batch
            avg_diff1_loss = total_diff1_loss / n_train_batch
            avg_diff2_loss = total_diff2_loss / n_train_batch

            log.log_epoch(
                {
                    "Learning Rate": optimizer.param_groups[0]["lr"],
                    "Train Loss": avg_loss,
                    "Train Data Loss": avg_data_loss,
                    "Train Phys Loss": avg_phys_loss,
                    "Train Kin Loss": avg_kin_loss,
                    "Train Diff1 Loss": avg_diff1_loss,
                    "Train Diff2 Loss": avg_diff2_loss,
                    "Train Data(Carbody)": float(data_loss_carbody.detach().item()),
                    "Train Data(Others)": float(data_loss_others.detach().item()),
                }
            )

        t_train = time.perf_counter() - t_epoch_start
        print(
            f"[Epoch {epoch + 1}] train done | total={avg_loss:.6e}, data={avg_data_loss:.6e}, "
            f"phys={avg_phys_loss:.6e}, kin={avg_kin_loss:.6e}, d1={avg_diff1_loss:.6e}, d2={avg_diff2_loss:.6e}, "
            f"time={t_train:.2f}s | "
            f"h2d={t_data_to_gpu:.2f}s, fwd={t_forward:.2f}s, bwd={t_backward:.2f}s, opt={t_optim:.2f}s"
        )

        t_val_start = time.perf_counter()
        with LaunchLogger("valid", epoch=epoch) as log:
            error = test_step(
                model,
                test_dataloader,
                epoch,
                device,
                SELECTED_CHANNEL_INDICES,
            )
            log.log_epoch({"Test MSE Loss": error})
        t_val = time.perf_counter() - t_val_start
        
        t_ckpt_start = time.perf_counter()
        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )
        t_ckpt = time.perf_counter() - t_ckpt_start

        t_epoch_total = time.perf_counter() - t_epoch_start
        print(
            f"[Epoch {epoch + 1}] summary | train={t_train:.2f}s, valid={t_val:.2f}s, "
            f"ckpt={t_ckpt:.2f}s, epoch_total={t_epoch_total:.2f}s"
        )

if __name__ == "__main__":
    main()