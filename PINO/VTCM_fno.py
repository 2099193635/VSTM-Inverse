'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-03-24 06:12:12
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-03-25 13:46:39
FilePath: /VTCM_PYTHON/PINO/VTCM_physicis_informed_fno.py
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
from omegaconf import DictConfig
from torch.utils.data import DataLoader   
from tqdm.auto import tqdm
from pino_utils import VTCMHDF5MapStyleDataset

# 目标输出通道（从原21通道中抽取）:
# [carbody_z, bogie_front_z, bogie_rear_z] x [disp, vel, acc]
SELECTED_CHANNEL_INDICES = [0, 1, 2, 7, 8, 9, 14, 15, 16]
POSITION_LABELS = ["Carbody (Z)", "Front Bogie (Z)", "Rear Bogie (Z)"]
COMPONENT_LABELS = ["Displacement", "Velocity", "Acceleration"]

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

            pred = model(input)
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

@hydra.main(version_base = "1.3", config_path = "conf", config_name='vtcm_physicis_informed_fno.yaml')
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
    train_h5 = "../datasets/VTCM_vertical/train_full_seq.hdf5"
    test_h5 = "../datasets/VTCM_vertical/train_full_seq.hdf5"
    validation_h5 = "../datasets/VTCM_vertical/validation_full_seq.hdf5"
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
    print(f"Selected output channels (from 21): {SELECTED_CHANNEL_INDICES}")
    print("Target outputs: Carbody/FrontBogie/RearBogie (Z) x [disp, vel, acc] => 9 channels")
    print("=" * 90)

    in_channels = train_dataset[0]["input"].shape[0]
    out_channels = len(SELECTED_CHANNEL_INDICES)
    model = FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    for epoch in range(cfg.max_epochs):
        t_epoch_start = time.perf_counter()
        total_loss = 0.0
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
                pred = model(input)
                loss = F.mse_loss(pred, output)
                t_forward += time.perf_counter() - t1

                t2 = time.perf_counter()
                loss.backward()
                t_backward += time.perf_counter() - t2

                t3 = time.perf_counter()
                optimizer.step()
                scheduler.step()
                t_optim += time.perf_counter() - t3

                total_loss += loss.item()

                if (batch_idx + 1) % 20 == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.3e}"})

            avg_loss = total_loss / len(train_dataloader)
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"], "Train Loss": avg_loss})

        t_train = time.perf_counter() - t_epoch_start
        print(
            f"[Epoch {epoch + 1}] train done | loss={avg_loss:.6e}, total={t_train:.2f}s | "
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