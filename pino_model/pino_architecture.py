from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import importlib

try:
	from importlib.util import find_spec
except Exception:
	find_spec = None

torch: Any = None
nn: Any = None
_TORCH_AVAILABLE = bool(find_spec and (find_spec("torch") is not None))

if _TORCH_AVAILABLE:
	torch = importlib.import_module("torch")
	nn = importlib.import_module("torch.nn")


class _BaseModule(nn.Module if _TORCH_AVAILABLE else object):
	pass


@dataclass
class PINOConfig:
	in_channels: int
	out_channels: int
	width: int = 64
	modes: int = 16
	n_layers: int = 4
	activation: str = "gelu"


class SpectralConv1d(_BaseModule):
	"""Minimal 1D spectral convolution used by PINO/FNO-style blocks."""

	def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
		if not _TORCH_AVAILABLE:
			raise RuntimeError("torch is required for SpectralConv1d")
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.modes = modes

		scale = 1.0 / (in_channels * out_channels)
		self.weights = nn.Parameter(
			scale
			* torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
		)

	def forward(self, x: Any) -> Any:
		# x: [B, C, T]
		batch_size, _, signal_len = x.shape
		x_ft = torch.fft.rfft(x, dim=-1)

		out_ft = torch.zeros(
			batch_size,
			self.out_channels,
			x_ft.size(-1),
			dtype=torch.cfloat,
			device=x.device,
		)

		m = min(self.modes, x_ft.size(-1))
		out_ft[:, :, :m] = torch.einsum(
			"bim,iom->bom", x_ft[:, :, :m], self.weights[:, :, :m]
		)

		return torch.fft.irfft(out_ft, n=signal_len, dim=-1)


class PINOBlock1d(_BaseModule):
	def __init__(self, width: int, modes: int, activation: str = "gelu") -> None:
		if not _TORCH_AVAILABLE:
			raise RuntimeError("torch is required for PINOBlock1d")
		super().__init__()
		self.spectral = SpectralConv1d(width, width, modes)
		self.w = nn.Conv1d(width, width, kernel_size=1)

		if activation == "relu":
			self.act = nn.ReLU()
		elif activation == "tanh":
			self.act = nn.Tanh()
		else:
			self.act = nn.GELU()

	def forward(self, x: Any) -> Any:
		return self.act(self.spectral(x) + self.w(x))


class PINOResidualHead(_BaseModule):
	"""
	输入: [B, T, C_in]
	输出: [B, T, C_out]
	典型用途：学习外激励导致的广义力/加速度残差。
	"""

	def __init__(self, config: PINOConfig) -> None:
		if not _TORCH_AVAILABLE:
			raise RuntimeError("torch is required for PINOResidualHead")
		super().__init__()
		self.config = config

		self.lift = nn.Conv1d(config.in_channels, config.width, kernel_size=1)
		self.blocks = nn.ModuleList(
			[
				PINOBlock1d(config.width, config.modes, config.activation)
				for _ in range(config.n_layers)
			]
		)
		self.proj = nn.Sequential(
			nn.Conv1d(config.width, config.width, kernel_size=1),
			nn.GELU(),
			nn.Conv1d(config.width, config.out_channels, kernel_size=1),
		)

	def forward(self, x: Any, mask: Optional[Any] = None) -> Any:
		# x: [B, T, C]
		if x.ndim != 3:
			raise ValueError(f"Expected input shape [B, T, C], got {tuple(x.shape)}")

		x = x.transpose(1, 2)  # [B, C, T]
		h = self.lift(x)
		for block in self.blocks:
			h = block(h)
		y = self.proj(h).transpose(1, 2)  # [B, T, C_out]

		if mask is not None:
			if mask.ndim == 2:
				mask = mask.unsqueeze(-1)
			y = y * mask
		return y

