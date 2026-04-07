from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class WindowConfig:
	window_size: int = 256
	stride: int = 64
	pred_horizon: int = 1
	use_context: bool = True


def _validate_1d_time_series(arr: np.ndarray, name: str) -> np.ndarray:
	if arr.ndim == 1:
		arr = arr[:, None]
	if arr.ndim != 2:
		raise ValueError(f"{name} must be [T, C] or [T], got shape {arr.shape}")
	return arr.astype(np.float32, copy=False)


class VTWindowDataset:
	"""
	Minimal dataset for forward/inverse learning.

	record format:
	- z: [T, Cz] state sequence (or response sequence)
	- u: [T, Cu] excitation sequence (optional for forward-only)
	- y: [T, Cy] measured output (optional for inverse)
	- c: [Cc] global context vector (optional)
	"""

	def __init__(self, records: Sequence[Dict[str, np.ndarray]], config: WindowConfig) -> None:
		self.records = list(records)
		self.config = config
		self.index_map: List[Tuple[int, int]] = []

		for ridx, rec in enumerate(self.records):
			z = _validate_1d_time_series(np.asarray(rec["z"]), "z")
			t_len = z.shape[0]
			max_start = t_len - config.window_size - config.pred_horizon
			if max_start < 0:
				continue
			for s in range(0, max_start + 1, config.stride):
				self.index_map.append((ridx, s))

	def __len__(self) -> int:
		return len(self.index_map)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		ridx, s = self.index_map[idx]
		rec = self.records[ridx]
		w = self.config.window_size
		h = self.config.pred_horizon

		z = _validate_1d_time_series(np.asarray(rec["z"]), "z")
		t_len = z.shape[0]
		z_in = z[s : s + w]          # [W, Cz]
		z_tgt = z[s + h : s + w + h] # [W, Cz], one-step aligned target by default

		sample: Dict[str, Any] = {
			"z_in": _to_tensor_or_array(z_in),
			"z_tgt": _to_tensor_or_array(z_tgt),
		}

		if "u" in rec and rec["u"] is not None:
			u = _validate_1d_time_series(np.asarray(rec["u"]), "u")
			sample["u_in"] = _to_tensor_or_array(u[s : s + w])

		if "y" in rec and rec["y"] is not None:
			y = _validate_1d_time_series(np.asarray(rec["y"]), "y")
			sample["y_in"] = _to_tensor_or_array(y[s : s + w])

		if self.config.use_context and "c" in rec and rec["c"] is not None:
			c = np.asarray(rec["c"], dtype=np.float32).reshape(-1)
			sample["context"] = _to_tensor_or_array(c)

		for k, v in rec.items():
			if k in {"z", "u", "y", "c"} or v is None:
				continue
			arr = np.asarray(v, dtype=np.float32)
			if arr.ndim >= 1 and arr.shape[0] == t_len:
				sample[k] = _to_tensor_or_array(arr[s + h : s + w + h])
			else:
				sample[k] = _to_tensor_or_array(arr)

		return sample


def build_dataloader(
	records: Sequence[Dict[str, np.ndarray]],
	config: Optional[WindowConfig] = None,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 0,
) -> Any:
	del num_workers  # kept for interface compatibility
	cfg = config or WindowConfig()
	ds = VTWindowDataset(records=records, config=cfg)
	return _SimpleBatchLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _to_tensor_or_array(x: np.ndarray) -> Any:
	return np.asarray(x, dtype=np.float32)


class _SimpleBatchLoader:
	def __init__(self, dataset: VTWindowDataset, batch_size: int, shuffle: bool) -> None:
		self.dataset = dataset
		self.batch_size = max(1, int(batch_size))
		self.shuffle = shuffle

	def __len__(self) -> int:
		n = len(self.dataset)
		return (n + self.batch_size - 1) // self.batch_size

	def __iter__(self):
		indices = np.arange(len(self.dataset))
		if self.shuffle:
			np.random.shuffle(indices)

		for i in range(0, len(indices), self.batch_size):
			batch_indices = indices[i : i + self.batch_size]
			items = [self.dataset[int(j)] for j in batch_indices]
			yield _stack_batch(items)


def _stack_batch(items: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
	keys = items[0].keys()
	out: Dict[str, np.ndarray] = {}
	for k in keys:
		out[k] = np.stack([np.asarray(it[k]) for it in items], axis=0)
	return out

