"""inverse_model/utils/dataset.py
HDF5 数据集与 batch 工具函数，供各模型训练脚本共用。
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class InverseH5Dataset(Dataset):
    """
    从 inverse_dataset_gen 导出的 HDF5 读取样本。
    输出 phys_z0 / phys_x 在 __init__ 中按全局 mean/std 归一化为 8 通道。

    Args:
        h5_path: HDF5 文件路径。
        out_norm_stats: 可传入训练集统计量 dict(mean, std) 供验证集复用；
                        传 None 时自动从当前数据计算（仅在训练集上调用）。
    """

    def __init__(self, h5_path: str | Path, out_norm_stats: dict | None = None):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            self.y       = np.asarray(f["y"],       dtype=np.float32)
            self.u       = np.asarray(f["u"],       dtype=np.float32)
            self.c       = np.asarray(f["c"],       dtype=np.float32)
            self.x_query = np.asarray(f["x_query"], dtype=np.float32)
            phys_z0_raw  = np.asarray(f["phys_z0"], dtype=np.float32)  # [N, T, 1]
            phys_x_raw   = np.asarray(f["phys_x"],  dtype=np.float32)  # [N, T, 7]
            self.phys_v  = np.asarray(f["phys_v"],  dtype=np.float32)  # [N, T, 7]
            self.phys_a  = np.asarray(f["phys_a"],  dtype=np.float32)  # [N, T, 7]
            self.vx      = np.asarray(f["vx"],      dtype=np.float32)
            self.seq_lengths = (
                np.asarray(f["seq_lengths"], dtype=np.int64) if "seq_lengths" in f else None
            )
            # 预存储各轮对 Z0（可选，用于提升 nexus 精度）
            self.phys_z0_w1 = np.asarray(f["phys_z0_w1"], dtype=np.float32) if "phys_z0_w1" in f else None
            self.phys_z0_w2 = np.asarray(f["phys_z0_w2"], dtype=np.float32) if "phys_z0_w2" in f else None
            self.phys_z0_w3 = np.asarray(f["phys_z0_w3"], dtype=np.float32) if "phys_z0_w3" in f else None
            self.phys_z0_w4 = np.asarray(f["phys_z0_w4"], dtype=np.float32) if "phys_z0_w4" in f else None

        # 拼合 8 通道输出 [N, T, 8]
        out_raw = np.concatenate([phys_z0_raw, phys_x_raw], axis=-1)

        if out_norm_stats is None:
            flat = out_raw.reshape(-1, out_raw.shape[-1])
            mean = flat.mean(axis=0)
            std  = flat.std(axis=0)
            std  = np.where(std < 1e-12, 1.0, std)
            self.out_norm_stats: dict = {"mean": mean, "std": std}
        else:
            self.out_norm_stats = out_norm_stats

        m = self.out_norm_stats["mean"]
        s = self.out_norm_stats["std"]
        self.out_norm = ((out_raw - m) / s).astype(np.float32)  # [N, T, 8]

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "y":        torch.from_numpy(self.y[idx]),
            "u":        torch.from_numpy(self.u[idx]),
            "c":        torch.from_numpy(self.c[idx]),
            "x_query":  torch.from_numpy(self.x_query[idx]),
            "out_norm": torch.from_numpy(self.out_norm[idx]),  # [T, 8]
            "phys_v":   torch.from_numpy(self.phys_v[idx]),
            "phys_a":   torch.from_numpy(self.phys_a[idx]),
            "vx":       torch.from_numpy(self.vx[idx]),
        }
        if self.seq_lengths is not None:
            sample["seq_lengths"] = torch.tensor(self.seq_lengths[idx], dtype=torch.long)
        for key in ("phys_z0_w1", "phys_z0_w2", "phys_z0_w3", "phys_z0_w4"):
            arr = getattr(self, key, None)
            if arr is not None:
                sample[key] = torch.from_numpy(arr[idx])
        return sample


def build_target(batch: dict, device: torch.device) -> torch.Tensor:
    """从 batch 中取归一化后的 8 通道输出目标，返回 [B, 8, T]。"""
    return batch["out_norm"].to(device, non_blocking=True).permute(0, 2, 1)
