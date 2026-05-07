"""inverse_model/utils/losses.py
通用损失函数，供各模型训练脚本共用。
"""
import numpy as np
import torch


def denormalize_output(pred_norm: torch.Tensor, out_norm_stats: dict,
                       device: torch.device) -> torch.Tensor:
    """将 [B, C, T] 归一化输出反归一化到物理量纲。"""
    mean = torch.as_tensor(out_norm_stats["mean"], device=device, dtype=pred_norm.dtype).view(1, -1, 1)
    std  = torch.as_tensor(out_norm_stats["std"],  device=device, dtype=pred_norm.dtype).view(1, -1, 1)
    return pred_norm * std + mean


def weighted_data_loss(pred: torch.Tensor, target: torch.Tensor,
                       z0_w: float = 5.0, w24_w: float = 2.0) -> torch.Tensor:
    """
    带通道权重的数据 MSE 损失，pred/target: [B, 8, T]。
    通道：[0]=Z0, [1]=Zc, [2]=Zt1, [3]=Zt2, [4]=Zw1, [5]=Zw2, [6]=Zw3, [7]=Zw4
    """
    w = pred.new_ones(8)
    w[0] = z0_w
    w[5] = w24_w
    w[7] = w24_w
    return ((pred - target) ** 2 * w.view(1, 8, 1)).mean()


def z0_spectral_loss(pred: torch.Tensor, target: torch.Tensor,
                     dt: float = 1e-4, hf_boost: float = 4.0,
                     hf_cutoff_hz: float = 2.4) -> torch.Tensor:
    """
    Z0 通道（ch=0）的 FFT 频谱幅值 MSE 损失，对高频段额外加权。
    pred/target: [B, C, T]
    """
    T = pred.shape[-1]
    amp_pred   = torch.fft.rfft(pred[:, 0, :],   dim=-1).abs()
    amp_target = torch.fft.rfft(target[:, 0, :], dim=-1).abs()
    freqs  = torch.fft.rfftfreq(T, d=dt).to(pred.device)
    freq_w = torch.where(freqs >= hf_cutoff_hz,
                         torch.full_like(freqs, hf_boost),
                         torch.ones_like(freqs))
    return ((amp_pred - amp_target) ** 2 * freq_w[None, :]).mean() / (T ** 2)
