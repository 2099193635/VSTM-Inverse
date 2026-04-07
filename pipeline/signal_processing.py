from __future__ import annotations

from typing import Tuple

import numpy as np


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
	if window <= 1:
		return x.copy()
	kernel = np.ones(window, dtype=np.float64) / float(window)
	if x.ndim == 1:
		return np.convolve(x, kernel, mode="same")
	return np.stack([np.convolve(col, kernel, mode="same") for col in x.T], axis=1)


def resample_uniform(
	t_src: np.ndarray,
	x_src: np.ndarray,
	dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Linear resampling onto uniform timeline."""
	t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
	x_src = np.asarray(x_src)
	if x_src.ndim == 1:
		x_src = x_src[:, None]

	t_new = np.arange(t_src[0], t_src[-1] + 0.5 * dt, dt)
	x_new = np.zeros((len(t_new), x_src.shape[1]), dtype=np.float64)
	for i in range(x_src.shape[1]):
		x_new[:, i] = np.interp(t_new, t_src, x_src[:, i])
	return t_new, x_new


def compute_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Simple FFT-based one-sided PSD."""
	x = np.asarray(x, dtype=np.float64)
	if x.ndim == 1:
		x = x[:, None]

	n = x.shape[0]
	freqs = np.fft.rfftfreq(n, d=1.0 / fs)
	x_demean = x - np.mean(x, axis=0, keepdims=True)
	spec = np.fft.rfft(x_demean, axis=0)
	psd = (np.abs(spec) ** 2) / (fs * max(n, 1))
	if n > 1:
		psd[1:-1] *= 2.0
	return freqs, psd


def band_energy_ratio(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
	freqs, psd = compute_psd(x, fs)
	total = np.sum(psd)
	if total <= 0:
		return 0.0
	mask = (freqs >= fmin) & (freqs <= fmax)
	return float(np.sum(psd[mask]) / total)


def spectral_l1(psd_pred: np.ndarray, psd_true: np.ndarray, eps: float = 1e-8) -> float:
	"""L1 distance in log-spectrum domain."""
	p = np.log(np.asarray(psd_pred, dtype=np.float64) + eps)
	t = np.log(np.asarray(psd_true, dtype=np.float64) + eps)
	return float(np.mean(np.abs(p - t)))

