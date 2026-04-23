import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from inverse_config import InverseConfig


logger = logging.getLogger(__name__)

@dataclass
class WindowConfig:
    window_size: int = 256     # time window length T
    stride: int = 128          # time window stride S
    sensor_indieces: list = None   # index of sensor DOFs, default to [carbody Z, front bogie Z, rear bogie Z]
    irr_indicies: list = None      # index of IRR DOFs, default to [vertical]

def compute_cond_vecotr(
        natural_freqs: np.ndarray,
        damping_ratios: np.ndarray,
        vx_mps: float,
        n_modal: int = 8,
) -> np.ndarray:

    freqs = np.zeros(n_modal, dtype=np.float32)
    damps = np.zeros(n_modal, dtype=np.float32)
    n = min(n_modal, len(natural_freqs))
    freqs[:n] = natural_freqs[:n]
    damps[:n] = damping_ratios[:n]
    return np.concatenate([freqs, damps, [vx_mps]], axis=0).astype(np.float32)

class InverseWindowDataset(Dataset):
    """
    Sliding window dataset, loaded from simulation_result_spatial.npz.

    Args:
        data_path: file path to the npz file containing the simulation results, it can be category, auto glob
        cfg: InverseConfig
        win_cfg: WindowConfig
        \
        val_ratio: ratio of validation data (divided by run, not window)
    """
    DEFAULT_SENSOR_DOFS = [2, 8, 14]    # vehicle Z=2, front bogie Z=8, rear bogie Z=14
    DEFAULT_IRR_DOFS = [0]              # only vertical

    def __int__(
        self,
        data_path: str | Path,
        cfg: InverseConfig,
        win_cfg: Optional[WindowConfig] = None,
        split: str = "train",
        val_ratio: float = 0.1        
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.win_cfg = win_cfg or WindowConfig()

        sensor_dofs = self.win_cfg.sensor_indieces or self.DEFAULT_SENSOR_DOFS
        irr_dofs = self.win_cfg.irr_indicies or self.DEFAULT_IRR_DOFS

        data_path = Path(data_path)
        if data_path.is_file():
            files = sorted(data_path.glob("*.npz"))
        else:
            files = [data_path]
        
        logger.info(f"Found {len(files)} npz files in {data_path}")

        # divided train/val by run, not window
        n_val = max(1, int(len(files) * val_ratio))
        if split == "train":
            files = files[n_val:]
        elif split in ("val", "test"):
            files = files[:n_val]
        
        self.records: list[dict] = []
        for f in files:
            self._load_file(f, sensor_dofs, irr_dofs)
        logger.info(f"[{split}] {len(self.records)} windows from {len(files)} runs")
    
    def _load_file(
            self, 
            path: Path, 
            sensor_dofs: list[int], 
            irr_dofs: list[int]
    ) -> None:
        data = np.load(path, allow_pickle=True)
        acc = data["acc"].astype(np.float32)          # (T_total, n_dofs)
        T_total = acc.shape[0]

        # Irregularity
        irr_list = []
        for d in irr_dofs:
            key = "irr_z" if d == 0 else "irr_y"
            irr_list.append(data[key].astype(np.float32))   # (T_total,)
        irr = np.stack(irr_list, axis=-1)    # (T_total, n_irr_dofs)

        # speed
        vx = float(np.mean(data["vx"])) if data["vx"].ndim == 0 else data["vx"].astype(np.float32)  # (T_total,)

        # Physically Derived Condition Vector
        n_freqs = data['natural_freqs'] if 'natural_freqs' in data else np.zeros(8)
        n_damps = data['damping_ratios'] if 'damping_ratios' in data else np.zeros(8)
        cond_vec = compute_cond_vecotr(n_freqs, n_damps, vx, self.cfg.n_modal)

        W = self.win_cfg.window_size
        S = self.win_cfg.stride

        for start in range(0, T_total - W + 1, S):
            end = start + W
            
            # acceleration window of sensor DOFs [T, n_sensors]
            y_win = acc[start:end,:][:, sensor_dofs]

            # Corresponding to spatial domain irregularities (aligned via x = v·t).
            x_start = int(start * vx * (1 / 200))  # assuming the sampling rate of 200 Hz.
            x_end = x_start + self.cfg.spatial_len
            if x_end > irr.shape[0]:
                break
            u_win = irr[x_start:x_end,:]  # [spatial_len, n_irr_dofs]
            
            x_query = np.linspace(0, 1, self.cfg.spatial_len, dtype = np.float32)

            self.records.append({
                "y":  y_win,        # [T, n_sensors]
                "u":  u_win,        # [L, n_dir]
                "c":  cond_vec,     # [n_cond]
                "x_query": x_query, # [L]
            })                          
    
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        return {k:torch.from_numpy(v) for k,v in rec.items()}


if __name__ == "__main__":
    cfg = InverseConfig()
    dataset = InverseWindowDataset("data/simulation_result_spatial.npz", cfg)
    print(dataset[0])