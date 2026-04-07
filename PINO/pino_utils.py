from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 物理参数 dataset 键名（匹配 dataset_vertical_builder.py 的命名规则）
_PHYSICS_PARAM_KEYS = (
    "vehicle_params",
    "rail_params",
    "fastener_params",
    "subrail_params",
    "line_params",
)


class VTCMHDF5MapStyleDataset(Dataset):
    """
    VTCM 时序 HDF5 数据集（map-style）。

    适配文件结构（如 datasets/VTCM_vertical/train.hdf5）：
      - input:          [N, 1, T]
      - output:         [N, C, T]
      - init_state:     [N, C]      (可选)
      - dt:             [N]         (可选)
      - source_file:    [N] utf-8   (可选)
      - window_start/window_end: [N] (可选)
      - vehicle_params: [N, p_veh]  (可选) 车辆参数，attrs["fields"] 记录列名
      - rail_params:    [N, p_rail] (可选) 钢轨参数
      - fastener_params:[N, p_fas]  (可选) 扣件参数
      - subrail_params: [N, p_sub]  (可选) 轨下结构参数
      - line_params:    [N, p_line] (可选) 线路几何参数
      - vx_mps:         [N]         (可选) 运行速度 m/s
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        preload_to_memory: bool = False,
        return_meta: bool = True,
    ):
        self.file_path = str(file_path)
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device
        self.preload_to_memory = preload_to_memory
        self.return_meta = return_meta

        with h5py.File(self.file_path, "r") as f:
            self.keys = list(f.keys())
            if "input" not in self.keys or "output" not in self.keys:
                raise KeyError(
                    f"HDF5 缺少必要键 input/output，当前键: {self.keys}"
                )
            self._length = int(f["input"].shape[0])

            # 读取各物理参数 dataset 的字段名（存于 dataset.attrs["fields"]）
            self.physics_field_names: Dict[str, List[str]] = {}
            for pk in _PHYSICS_PARAM_KEYS:
                if pk in f:
                    fields_str = f[pk].attrs.get("fields", "")
                    self.physics_field_names[pk] = (
                        fields_str.split(",") if fields_str else []
                    )

        self._cache: Optional[Dict[str, np.ndarray]] = None
        if self.preload_to_memory:
            self._cache = self._load_all_arrays()

    def _load_all_arrays(self) -> Dict[str, np.ndarray]:
        data: Dict[str, np.ndarray] = {}
        with h5py.File(self.file_path, "r") as f:
            for k in self.keys:
                data[k] = np.array(f[k])
        return data

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _to_device_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(arr).to(device)

    def _get_raw_item(self, idx: int) -> Dict[str, np.ndarray]:
        if self._cache is not None:
            return {k: self._cache[k][idx] for k in self.keys}

        out: Dict[str, np.ndarray] = {}
        with h5py.File(self.file_path, "r") as f:
            for k in self.keys:
                out[k] = np.array(f[k][idx])
        return out

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, int, str]]:
        raw = self._get_raw_item(idx)

        sample: Dict[str, Union[torch.Tensor, float, int, str]] = {
            "input": self._to_device_tensor(raw["input"].astype(np.float32), self.device),
            "output": self._to_device_tensor(raw["output"].astype(np.float32), self.device),
        }

        if "init_state" in raw:
            sample["init_state"] = self._to_device_tensor(
                raw["init_state"].astype(np.float32), self.device
            )

        if not self.return_meta:
            return sample

        if "dt" in raw:
            sample["dt"] = float(raw["dt"])
        if "window_start" in raw:
            sample["window_start"] = int(raw["window_start"])
        if "window_end" in raw:
            sample["window_end"] = int(raw["window_end"])
        if "source_file" in raw:
            src = raw["source_file"]
            if isinstance(src, (bytes, np.bytes_)):
                sample["source_file"] = src.decode("utf-8", errors="ignore")
            else:
                sample["source_file"] = str(src)

        # 物理参数张量（vehicle/rail/fastener/subrail/line + vx_mps）
        for pk in _PHYSICS_PARAM_KEYS:
            if pk in raw:
                sample[pk] = self._to_device_tensor(
                    np.atleast_1d(raw[pk]).astype(np.float32), self.device
                )
        if "vx_mps" in raw:
            sample["vx_mps"] = float(raw["vx_mps"])

        return sample


def create_vtcm_hdf5_dataloader(
    file_path: Union[str, Path],
    batch_size: int = 4,
    shuffle: bool = True,
    device: Union[str, torch.device] = "cpu",
    preload_to_memory: bool = False,
    return_meta: bool = True,
) -> DataLoader:
    dataset = VTCMHDF5MapStyleDataset(
        file_path=file_path,
        device=device,
        preload_to_memory=preload_to_memory,
        return_meta=return_meta,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


if __name__ == "__main__":
    train_h5 = "/workspace/VTCM_PYTHON/datasets/VTCM_vertical/train.hdf5"
    ds = VTCMHDF5MapStyleDataset(train_h5, device="cpu", preload_to_memory=False)
    sample0 = ds[0]
    print("keys:", list(sample0.keys()))
    print("input:", tuple(sample0["input"].shape))
    print("output:", tuple(sample0["output"].shape))
    if "init_state" in sample0:
        print("init_state:", tuple(sample0["init_state"].shape))
    print()
    print("physics_field_names:", ds.physics_field_names)
    for pk in _PHYSICS_PARAM_KEYS:
        if pk in sample0:
            print(f"{pk}: shape={tuple(sample0[pk].shape)}")
    if "vx_mps" in sample0:
        print(f"vx_mps: {sample0['vx_mps']:.2f} m/s")
