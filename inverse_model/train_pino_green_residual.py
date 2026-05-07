"""
train_pino_green_residual.py
================================================================================
Physics-Guided Residual FNO  —  VTCM **Inverse** Problem
  Vehicle 21-DOF response y(t) → Rail irregularity u(t)

Based on Green's function feasibility analysis (test_green_function.py).
Two strategies for embedding the identified FRF as physical prior:

  Strategy A  ─ Green's Function Inverse Baseline  (--mode residual | both)
    In forward problem:  Y_dof(ω) = H_dof(ω) · U(ω)
    Wiener-type inverse: Û(ω) = Σ_dof  H_dof*(ω)·M_dof(ω)·Y_dof(ω)
                                  ─────────────────────────────────────
                                  Σ_dof  |H_dof(ω)|²·M_dof(ω)  +  reg

    u_pred = GreenInverseBaseline(y) + FNO_θ(y)
             ↑ linear physics inverse           ↑ non-linear residual

    Modal decoupling:
      u_sym  ← Z/Y DOFs  (symmetric excitation → vertical motion)
      u_asym ← Roll DOFs (anti-symmetric       → rolling motion)
      u_L = u_sym + u_asym/2
      u_R = u_sym - u_asym/2

  Strategy B  ─ Forward FRF Consistency Loss  (--mode frf_loss | both)
    After predicting û(t), re-apply the identified FRF forward and
    compare with the observed response in high-coherence bands only:

    L_frf = Σ_dof Σ_ω  M_dof(ω) · |H_dof(ω)·Û(ω) − Y_dof(ω)|² / Σ M_dof(ω)

    M_dof(ω) = 1  if  MSC_dof(ω) > msc_threshold  else  0
    → Enforces physical self-consistency in linearly-valid frequency bands.
      Prevents the FNO from hallucinating irregularities that would
      produce a y inconsistent with the identified dynamics.

CLI --mode choices:
  baseline   ─ vanilla FNO  (pure data-driven, no physics)
  residual   ─ Strategy A only
  frf_loss   ─ Strategy B only
  both       ─ Strategy A + B  [recommended]

Dataset:  datasets/VTCM_vertical/{train,validation}_full_seq.hdf5
  input  : [N, 2, T]   ─ left/right rail absolute elevation   → LABEL  u
  output : [N, 21, T]  ─ vehicle 21-DOF absolute displacement → INPUT  y
  dt     : [N]         ─ time step per sample (s)
  vx_mps : [N]         ─ speed per sample (m/s)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_DIR  = Path(__file__).parent
_ROOT = _DIR.parent
sys.path.insert(0, str(_ROOT / "physicsnemo"))
sys.path.insert(0, str(_DIR))

from physicsnemo.models.fno import FNO
from physicsnemo.utils.checkpoint import save_checkpoint

# ── DOF metadata ──────────────────────────────────────────────────────────────
# VTCM_inverse dataset: 1 sensor channel (vehicle body Z)  →  1 vertical irregularity
DOF_LABELS = ["Body-Z"]
N_DOF   = 1    # sensor channels in y
N_RAIL  = 1    # irregularity channels in u (vertical only)

# Fixed time step for VTCM simulations
_DEFAULT_DT = 1e-4   # s  (fs = 10 000 Hz)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset  (inverse: input=y [1,T], label=u [1,T])
# ═══════════════════════════════════════════════════════════════════════════════

class VTCMInverseDataset(Dataset):
    """Inverse dataset for VTCM_inverse.

    HDF5 layout (channel-last):
        y       : [N, T, 1]  – vehicle sensor response (Body-Z)
        u       : [N, T, 1]  – vertical rail irregularity
        c       : [N, 17]    – conditioning: 8 nat.freqs + 8 damp.ratios + vx
        x_query : [N, T]     – normalised spatial position [0, 1]
        vx      : [N, 1]     – speed (m/s)

    FNO input  (x) : y_bp [1, T]  – bandpass vehicle response
    FNO target (u) : u_bp [1, T]  – bandpass vertical rail irregularity
    """
    BP_LOW  = 0.5    # Hz
    BP_HIGH = 50.0   # Hz

    def __init__(self, h5_path: str | Path, dt: float = _DEFAULT_DT,
                 seg_len: int | None = None, augment: bool = False):
        """
        seg_len  : if set, each __getitem__ returns a random sub-segment of
                   this length (in samples) instead of the full sequence.
                   Greatly reduces overfitting when training set is small.
        augment  : if True (training mode), the start offset is sampled
                   randomly; if False (val mode), the centre segment is used.
        """
        self.h5_path = Path(h5_path)
        self.dt      = dt
        self.seg_len = seg_len
        self.augment = augment
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            # channel-last [N, T, 1]
            self._y  = np.asarray(f["y"],  dtype=np.float32)  # [N, T, 1]
            self._u  = np.asarray(f["u"],  dtype=np.float32)  # [N, T, 1]
            self._vx = np.asarray(f["vx"], dtype=np.float32)  # [N, 1]
            self._c  = np.asarray(f["c"],  dtype=np.float32)  # [N, 17]

    def __len__(self) -> int:
        return self._y.shape[0]

    def _sos(self):
        fs  = 1.0 / self.dt
        nyq = fs / 2.0
        return sp_signal.butter(
            4,
            [self.BP_LOW / nyq, min(self.BP_HIGH / nyq, 0.99)],
            btype="band", output="sos",
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sos = self._sos()
        bp  = lambda x: sp_signal.sosfiltfilt(sos, x).astype(np.float32)

        # ── Vehicle response (FNO input) ──────────────────────────────────────
        y_bp = bp(self._y[idx, :, 0])      # [T_full]
        u_bp = bp(self._u[idx, :, 0])      # [T_full]

        # ── Random / fixed sub-segment cropping (data augmentation) ──────────
        T_full = y_bp.shape[0]
        seg = self.seg_len
        if seg is not None and seg < T_full:
            if self.augment:
                start = np.random.randint(0, T_full - seg + 1)
            else:
                start = (T_full - seg) // 2   # deterministic centre segment
            y_bp = y_bp[start: start + seg]
            u_bp = u_bp[start: start + seg]

        x_in    = y_bp[np.newaxis, :]      # [1, T]
        # ── Rail irregularity (inverse label) ─────────────────────────────────
        u_label = u_bp[np.newaxis, :]      # [1, T]

        return {
            "x":  torch.from_numpy(x_in),      # [1, T]  – FNO input
            "u":  torch.from_numpy(u_label),   # [1, T]  – inverse target
            "dt": torch.tensor(self.dt, dtype=torch.float32),
            "vx": torch.tensor(float(self._vx[idx, 0]), dtype=torch.float32),
            "c":  torch.from_numpy(self._c[idx]),   # [17]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Pre-training FRF Identification
# ═══════════════════════════════════════════════════════════════════════════════

class FRFIdentifier:
    """
    Welch cross-spectrum FRF identification.
    Forward model:  Y_dof(ω) = H_dof(ω) · U_modal(ω)

    After run():
      self.H_emp   : [N_DOF, F]  complex   – averaged FRF
      self.msc     : [N_DOF, F]  float32   – averaged MSC
      self.msc_mask: [N_DOF, F]  float32   – 1 where MSC >= threshold
      self.freqs   : [F]         float32   – Hz
    """

    def __init__(self, dt: float, n_avg: int = 8, msc_threshold: float = 0.5):
        self.dt            = dt
        self.n_avg         = n_avg
        self.msc_threshold = msc_threshold

    def _welch_frf(self, u, y):
        T       = len(u)
        nperseg = max(512, T // self.n_avg)
        fs      = 1.0 / self.dt
        f, Suu  = sp_signal.welch(u, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window="hann")
        _, Syu  = sp_signal.csd(u, y,  fs=fs, nperseg=nperseg, noverlap=nperseg//2, window="hann")
        H = Syu / (Suu + 1e-6 * np.max(np.abs(Suu)))
        return f, H

    def _welch_coh(self, u, y):
        T       = len(u)
        nperseg = max(512, T // self.n_avg)
        fs      = 1.0 / self.dt
        f, coh  = sp_signal.coherence(u, y, fs=fs, nperseg=nperseg,
                                      noverlap=nperseg//2, window="hann")
        return f, coh

    def run(self, dataset: VTCMInverseDataset, n_samples: Optional[int] = None,
            train_split: float = 0.6) -> None:
        n_samples = min(n_samples or len(dataset), len(dataset))
        print(f"\n[FRF Identification]  samples={n_samples}, n_avg={self.n_avg}, "
              f"msc_threshold={self.msc_threshold}")

        H_acc   = None
        coh_acc = None
        n_valid = np.zeros(N_DOF, dtype=int)
        freqs_ref = None

        for idx in tqdm(range(n_samples), desc="  FRF", leave=False):
            item  = dataset[idx]
            T     = item["x"].shape[1]
            n_tr  = int(T * train_split)

            for dof in range(N_DOF):
                y_tr = item["x"][dof, :n_tr].numpy()
                u_tr = item["u"][0, :n_tr].numpy()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f, H   = self._welch_frf(u_tr, y_tr)
                        _, coh = self._welch_coh(u_tr, y_tr)
                    if H_acc is None:
                        F_len   = len(f)
                        H_acc   = np.zeros((N_DOF, F_len), dtype=complex)
                        coh_acc = np.zeros((N_DOF, F_len), dtype=float)
                        freqs_ref = f
                    F_use = min(len(f), H_acc.shape[1])
                    H_acc[dof,   :F_use] += H[:F_use]
                    coh_acc[dof, :F_use] += coh[:F_use]
                    n_valid[dof] += 1
                except Exception:
                    pass

        for dof in range(N_DOF):
            if n_valid[dof] > 0:
                H_acc[dof]   /= n_valid[dof]
                coh_acc[dof] /= n_valid[dof]

        self.freqs    = freqs_ref.astype(np.float32)
        self.H_emp    = H_acc                          # [N_DOF, F]
        self.msc      = coh_acc.astype(np.float32)
        self.msc_mask = (coh_acc >= self.msc_threshold).astype(np.float32)

        avg_msc  = float(np.mean(coh_acc))
        n_active = int(np.sum(self.msc_mask.sum(axis=1) > 5))
        print(f"  Done.  F_bins={len(freqs_ref)}, avg_MSC={avg_msc:.3f}, "
              f"DOFs with >5 active bins: {n_active}/{N_DOF}")
        for dof in range(N_DOF):
            print(f"    {DOF_LABELS[dof]:<14}  avg_MSC={coh_acc[dof].mean():.3f}  "
                  f"mask_bins={int(self.msc_mask[dof].sum())}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Strategy A — Green's Function Inverse Baseline (Wiener Filter)
# ═══════════════════════════════════════════════════════════════════════════════

class GreenInverseBaseline(nn.Module):
    """
    Linear inverse of the identified FRF via Wiener matched-filter estimator.

    Forward: Y_dof(ω) = H_dof(ω) · U_modal(ω)
    Inverse: Û_modal(ω) = Σ_dof  H_dof*(ω) · M_dof(ω) · Y_dof(ω)
                           ─────────────────────────────────────────
                           Σ_dof  |H_dof(ω)|² · M_dof(ω)  +  reg

    Summation for u_sym  uses Z/Y DOFs (symmetric excitation).
    Summation for u_asym uses Roll DOFs (anti-symmetric excitation).

    All buffers move to GPU automatically.  No gradient passes through.
    """

    def __init__(self, H_emp: np.ndarray, msc_mask: np.ndarray,
                 dt: float, f_max: float = 50.0, reg: float = 1e-3):
        super().__init__()
        self.dt  = dt
        self.reg = reg
        F_len    = H_emp.shape[1]
        fs       = 1.0 / dt
        f_axis   = np.linspace(0, fs / 2, F_len)
        f_mask   = (f_axis <= f_max).astype(np.float32)   # [F]
        mask     = msc_mask * f_mask.reshape(1, -1)       # [N_DOF, F]

        self.register_buffer("H_real",    torch.from_numpy(H_emp.real.astype(np.float32)))
        self.register_buffer("H_imag",    torch.from_numpy(H_emp.imag.astype(np.float32)))
        self.register_buffer("mask",      torch.from_numpy(mask))
        self.all_dofs = list(range(N_DOF))

    @property
    def H(self) -> torch.Tensor:
        return torch.complex(self.H_real, self.H_imag)   # [N_DOF, F]

    @torch.no_grad()
    def _wiener_inv(self, Y_fft: torch.Tensor, dof_indices: list) -> torch.Tensor:
        """
        Y_fft  : [B, N_DOF, F_model]  complex
        Returns: [B, T]  Wiener inverse time-domain estimate
        """
        B, _, F_model = Y_fft.shape
        F_h   = self.H_real.shape[1]
        F_use = min(F_model, F_h)

        H_sel    = self.H[:, :F_use]           # [N_DOF, F_use] complex
        mask_sel = self.mask[:, :F_use]        # [N_DOF, F_use] float

        numer = torch.zeros(B, F_use, dtype=torch.complex64, device=Y_fft.device)
        denom = torch.zeros(B, F_use, dtype=torch.float32,   device=Y_fft.device)

        for dof in dof_indices:
            H_d = H_sel[dof]                   # [F_use] complex
            M_d = mask_sel[dof]                # [F_use] float
            Y_d = Y_fft[:, dof, :F_use]        # [B, F_use] complex
            numer = numer + torch.conj(H_d).unsqueeze(0) * M_d.unsqueeze(0) * Y_d
            denom = denom + (H_d.real**2 + H_d.imag**2) * M_d.unsqueeze(0)

        reg_val = self.reg * (denom.max(dim=-1, keepdim=True).values + 1e-12)
        U_hat   = numer / (denom + reg_val)    # [B, F_use] complex

        U_pad   = torch.zeros(B, F_model, dtype=torch.complex64, device=Y_fft.device)
        U_pad[:, :F_use] = U_hat
        T       = (F_model - 1) * 2
        return torch.fft.irfft(U_pad, n=T)    # [B, T]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : [B, 1, T]  – vehicle sensor response
        Returns: [B, 1, T]  – vertical irregularity linear inverse estimate
        """
        B, _, T  = x.shape
        Y_fft    = torch.fft.rfft(x, dim=-1)   # [B, 1, F_model]
        u_t      = self._wiener_inv(Y_fft, self.all_dofs)[..., :T]  # [B, T]
        return u_t.unsqueeze(1)                # [B, 1, T]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Strategy B — Forward FRF Consistency Loss
# ═══════════════════════════════════════════════════════════════════════════════

class ForwardFRFConsistencyLoss(nn.Module):
    """
    Given predicted u_pred [B, 2, T]:
      Re-apply the identified forward FRF and compare with observed y
      in high-MSC frequency bands only.

      L = Σ_dof Σ_ω  M_dof(ω) · |H_dof(ω)·Û_dof(ω) − Y_dof(ω)|² / Σ M_dof(ω)

    u_pred[:, 0] = u_sym  → used for Z/Y DOFs
    u_pred[:, 1] = u_asym → used for Roll DOFs
    """

    def __init__(self, H_emp: np.ndarray, msc_mask: np.ndarray,
                 dt: float, f_max: float = 50.0):
        super().__init__()
        self.dt = dt
        F_len   = H_emp.shape[1]
        fs      = 1.0 / dt
        f_axis  = np.linspace(0, fs / 2, F_len)
        f_msk   = (f_axis <= f_max).astype(np.float32)
        mask    = msc_mask * f_msk.reshape(1, -1)   # [N_DOF, F]

        self.register_buffer("H_real", torch.from_numpy(H_emp.real.astype(np.float32)))
        self.register_buffer("H_imag", torch.from_numpy(H_emp.imag.astype(np.float32)))
        self.register_buffer("mask",   torch.from_numpy(mask))

    @property
    def H(self) -> torch.Tensor:
        return torch.complex(self.H_real, self.H_imag)

    def forward(self, u_pred: torch.Tensor,
                x_obs:  torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        u_pred : [B, 1, T]   – predicted vertical irregularity
        x_obs  : [B, 1, T]  – observed vehicle response
        Returns: scalar loss, per-dof loss dict
        """
        B, _, T  = u_pred.shape
        F_model  = T // 2 + 1
        F_h      = self.H_real.shape[1]
        F_use    = min(F_model, F_h)

        U_fft     = torch.fft.rfft(u_pred[:, 0, :], dim=-1)[:, :F_use]  # [B, F_use]
        Y_obs_fft = torch.fft.rfft(x_obs[:, 0, :], dim=-1)[:, :F_use]   # [B, F_use]
        H_use    = self.H[0, :F_use]     # [F_use] complex
        mask_use = self.mask[0, :F_use]  # [F_use] float

        per_dof = {}
        if mask_use.sum() < 1.0:
            return torch.tensor(0.0, device=u_pred.device), per_dof

        Y_hat      = H_use.unsqueeze(0) * U_fft           # [B, F_use]
        diff       = Y_hat - Y_obs_fft
        err        = (diff.real**2 + diff.imag**2) * mask_use.unsqueeze(0)
        total_loss = err.sum() / (mask_use.sum() * B)
        per_dof[0] = float(total_loss.item())
        return total_loss, per_dof


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Physics Forward Loss  (Strategy C)
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsForwardLoss(nn.Module):
    """
    Strategy C — Physics Forward Reconstruction Loss

    Given predicted u_pred [B, 1, T], apply the identified forward FRF
    to reconstruct the vehicle response in **time domain**, then compare
    with the observed y [B, 1, T] using MSE.

      Y_hat(ω) = H(ω) · U_pred(ω)
      y_hat(t) = iFFT[ Y_hat(ω) ]
      L_phys   = MSE( y_hat(t), y_obs(t) )

    This is a differentiable physics-consistency loss: gradients flow back
    through irfft / rfft / FNO parameters.
    Inspired by the forward VTCM solver in generate_main.py:
      irregularity u(t)  →  [contact + suspension dynamics]  →  response y(t)
    Here the linear FRF H(ω) approximates that forward map.
    """

    def __init__(self, H_emp: np.ndarray, dt: float, f_max: float = 50.0,
                 win_len: int = 4096):
        """
        H_emp   : [N_DOF, F]  complex  – empirically identified FRF
        dt      : time step (s)
        f_max   : upper frequency limit (Hz)
        win_len : window length (samples) used for FFT to save GPU memory.
                  When the sequence T > win_len, a random window is cropped.
                  Default 4096 (~0.41 s at fs=10 kHz).
        """
        super().__init__()
        self.dt      = dt
        self.win_len = win_len
        F_len   = H_emp.shape[1]
        fs      = 1.0 / dt
        f_axis  = np.linspace(0, fs / 2, F_len)
        f_win   = (f_axis <= f_max).astype(np.float32)  # [F]

        self.register_buffer("H_real", torch.from_numpy(H_emp[0].real.astype(np.float32)))
        self.register_buffer("H_imag", torch.from_numpy(H_emp[0].imag.astype(np.float32)))
        self.register_buffer("f_win",  torch.from_numpy(f_win))

    @property
    def H(self) -> torch.Tensor:
        return torch.complex(self.H_real, self.H_imag)   # [F]

    def forward(self, u_pred: torch.Tensor,
                y_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        u_pred : [B, 1, T]  – predicted vertical irregularity
        y_obs  : [B, 1, T]  – observed vehicle response (FNO input)
        Returns:
            loss   : scalar MSE in time domain
            y_hat  : [B, 1, W] reconstructed response window (for diagnostics)
        """
        B, _, T  = u_pred.shape
        W = min(self.win_len, T)

        # Random window crop to limit GPU memory usage
        if T > W:
            start  = torch.randint(0, T - W + 1, (1,)).item()
            u_win  = u_pred[:, 0, start: start + W]   # [B, W]
            y_win  = y_obs[:, 0, start: start + W]    # [B, W]
        else:
            u_win  = u_pred[:, 0, :]                  # [B, T]
            y_win  = y_obs[:, 0, :]

        F_model  = W // 2 + 1
        F_h      = self.H_real.shape[0]
        F_use    = min(F_model, F_h)

        # Forward: U → Y_hat in frequency domain
        U_fft   = torch.fft.rfft(u_win, dim=-1)             # [B, F_model]
        H_use   = self.H[:F_use]                             # [F_use] complex
        win_use = self.f_win[:F_use]                         # [F_use] float

        Y_hat_fft = torch.zeros(B, F_model, dtype=torch.complex64,
                                device=u_pred.device)
        Y_hat_fft[:, :F_use] = U_fft[:, :F_use] * H_use.unsqueeze(0) * win_use.unsqueeze(0)

        # Back to time domain
        y_hat = torch.fft.irfft(Y_hat_fft, n=W)             # [B, W]
        y_hat = y_hat.unsqueeze(1)                           # [B, 1, W]

        loss = F.mse_loss(y_hat, y_win.unsqueeze(1))
        return loss, y_hat


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Residual Inverse FNO
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualInverseFNO(nn.Module):
    """
    Inverse model:  x [B, 21, T]  →  u [B, 2, T]

    u_pred = GreenInverseBaseline(x)  +  FNO_θ(x)
             ↑ no-gradient Wiener inv    ↑ non-linear residual

    When green_inv is None, reduces to vanilla FNO (--mode baseline).
    """

    def __init__(self, fno: FNO, green_inv: Optional[GreenInverseBaseline] = None):
        super().__init__()
        self.fno       = fno
        self.green_inv = green_inv
        self.use_green = (green_inv is not None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns: (u_pred [B,2,T],  u_green [B,2,T] or None)
        """
        u_fno = self.fno(x)   # [B, 2, T]

        if self.use_green:
            u_green = self.green_inv(x)      # [B, 2, T]  (detached / no grad)
            u_pred  = u_green + u_fno
        else:
            u_green = None
            u_pred  = u_fno

        return u_pred, u_green


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Visualisation / Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════════

_PLOT_RC = {
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  8,
    "lines.linewidth":  1.4,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "figure.dpi":       150,
    "savefig.dpi":      300,
}
_CH_NAMES = ["u_vertical"]


def _save_val_fig(pred, target, epoch, fig_dir, u_green=None):
    """pred / target / u_green : [2, T]"""
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update(_PLOT_RC)
    T  = pred.shape[1]
    t  = np.arange(T)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes = axes.reshape(1, 2)  # keep consistent indexing as axes[ch, col]

    for ch in range(N_RAIL):
        ax_t = axes[ch, 0]
        ax_f = axes[ch, 1]

        ax_t.plot(t, target[ch], color="#1B4F72", lw=0.9, alpha=0.85, label="Ground truth")
        if u_green is not None:
            ax_t.plot(t, u_green[ch], color="#7D3C98", lw=0.8, ls=":", alpha=0.75,
                      label="Linear inverse (Green)")
        ax_t.plot(t, pred[ch], color="#C0392B", lw=0.9, ls="--", alpha=0.90,
                  label=f"Residual FNO  ρ={np.corrcoef(pred[ch], target[ch])[0,1]:.3f}")
        ax_t.set_title(f"(ch{ch}) {_CH_NAMES[ch]} — Time domain")
        ax_t.set_xlabel("Time step"); ax_t.set_ylabel("Amplitude (m)")
        if ch == 0:
            ax_t.legend(loc="upper right", fontsize=7)

        from scipy.signal import welch
        f_t, ps_t = welch(target[ch], nperseg=min(512, T//4))
        f_p, ps_p = welch(pred[ch],   nperseg=min(512, T//4))
        ax_f.semilogy(f_t, ps_t, color="#1B4F72", lw=1.0, alpha=0.85, label="GT")
        ax_f.semilogy(f_p, ps_p, color="#C0392B", lw=1.0, ls="--", alpha=0.90, label="Pred")
        if u_green is not None:
            f_g, ps_g = welch(u_green[ch], nperseg=min(512, T//4))
            ax_f.semilogy(f_g, ps_g, color="#7D3C98", lw=0.8, ls=":", alpha=0.75, label="Green")
        ax_f.set_title(f"(ch{ch}) {_CH_NAMES[ch]} — PSD")
        ax_f.set_xlabel("Norm. freq."); ax_f.set_ylabel("PSD")
        ax_f.legend(loc="upper right", fontsize=7)

    fig.suptitle(f"Inverse Prediction @ Epoch {epoch+1}", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(fig_dir, f"val_ep{epoch:04d}.png"), bbox_inches="tight")
    plt.close(fig)


def _metrics(pred, target):
    out = {}
    for ch in range(pred.shape[0]):
        p, t = pred[ch], target[ch]
        rms_p = np.sqrt(np.mean(p**2)) + 1e-12
        rms_t = np.sqrt(np.mean(t**2)) + 1e-12
        p_sc  = p * (rms_t / rms_p)
        var_t = float(np.var(t)) + 1e-20
        out[f"r2_norm_ch{ch}"] = 1.0 - float(np.mean((p_sc - t)**2)) / var_t
        out[f"corr_ch{ch}"]    = float(np.corrcoef(p, t)[0, 1])
        out[f"rmse_ch{ch}"]    = float(np.sqrt(np.mean((p - t)**2)))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Validation loop
# ═══════════════════════════════════════════════════════════════════════════════

def val_epoch(model, loader, frf_loss_fn, physics_loss_fn, device,
              lambda_frf, lambda_physics,
              epoch, fig_dir, vis_every):
    model.eval()
    s_loss = s_data = s_frf = s_phys = 0.0
    n_batch = 0
    viz_pred = viz_target = viz_green = None

    with torch.no_grad():
        for batch in loader:
            x      = batch["x"].to(device)
            target = batch["u"].to(device)

            u_pred, u_green = model(x)
            loss_data = F.mse_loss(u_pred, target)
            loss_frf  = torch.tensor(0.0, device=device)
            loss_phys = torch.tensor(0.0, device=device)

            if frf_loss_fn is not None:
                loss_frf, _ = frf_loss_fn(u_pred, x)
            if physics_loss_fn is not None:
                loss_phys, _ = physics_loss_fn(u_pred, x)

            loss     = loss_data + lambda_frf * loss_frf + lambda_physics * loss_phys
            s_loss  += loss.item()
            s_data  += loss_data.item()
            s_frf   += loss_frf.item()
            s_phys  += loss_phys.item()
            n_batch += 1

            if viz_pred is None:
                viz_pred   = u_pred[0].cpu().numpy()
                viz_target = target[0].cpu().numpy()
                if u_green is not None:
                    viz_green = u_green[0].cpu().numpy()

    n_batch = max(n_batch, 1)
    if viz_pred is not None and (epoch % vis_every == 0):
        _save_val_fig(viz_pred, viz_target, epoch, fig_dir, viz_green)

    m = _metrics(viz_pred, viz_target) if viz_pred is not None else {}
    return dict(loss=s_loss/n_batch, data=s_data/n_batch,
                frf=s_frf/n_batch, phys=s_phys/n_batch, **m)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Physics-Guided Residual FNO — VTCM Inverse (y→u)"
    )
    p.add_argument("--dataset_dir",        type=str, default=str(_ROOT / "datasets" / "VTCM_inverse"))
    p.add_argument("--train_suffix",       type=str, default="_full_seq")
    p.add_argument("--val_suffix",         type=str, default="_full_seq")
    p.add_argument("--dt",                 type=float, default=_DEFAULT_DT,
                   help="Time step in seconds (default 1e-4 for VTCM simulations)")
    p.add_argument("--latent_channels",    type=int,   default=32)
    p.add_argument("--num_fno_layers",     type=int,   default=4)
    p.add_argument("--num_fno_modes",      type=int,   default=12)
    p.add_argument("--decoder_layers",     type=int,   default=2)
    p.add_argument("--decoder_layer_size", type=int,   default=64)
    p.add_argument("--padding",            type=int,   default=9)
    p.add_argument("--epochs",             type=int,   default=200)
    p.add_argument("--batch_size",         type=int,   default=16)
    p.add_argument("--lr",                 type=float, default=3e-3)
    p.add_argument("--weight_decay",       type=float, default=1e-3)
    p.add_argument("--seg_len",            type=int,   default=0,
                   help="Sub-segment length for training data augmentation "
                        "(samples). 0 = use full sequence (default).")
    p.add_argument("--early_stop_patience", type=int,  default=30,
                   help="Stop training if val loss does not improve for this "
                        "many epochs. 0 = disabled.")
    p.add_argument("--lr_T_max",           type=int,   default=200)
    p.add_argument("--lr_eta_min",         type=float, default=1e-5)
    p.add_argument("--clip_grad",          type=float, default=1.0)
    p.add_argument("--mode", type=str, default="frf_loss",
                   choices=["baseline", "residual", "frf_loss", "both"])
    p.add_argument("--lambda_frf",         type=float, default=0.02)
    p.add_argument("--lambda_physics",     type=float, default=0.1,
                   help="Weight of PhysicsForwardLoss (time-domain FRF reconstruction). "
                        "0 = disabled. Requires FRF identification (needs training data).")
    p.add_argument("--phys_win_len",       type=int,   default=4096,
                   help="FFT window length for PhysicsForwardLoss. "
                        "Smaller = less GPU memory. Default=4096 (~0.41s at 10kHz).")
    p.add_argument("--msc_threshold",      type=float, default=0.50)
    p.add_argument("--wiener_reg",         type=float, default=1e-3)
    p.add_argument("--frf_n_avg",          type=int,   default=8)
    p.add_argument("--frf_samples",        type=int,   default=None)
    p.add_argument("--frf_train_split",    type=float, default=0.6)
    p.add_argument("--ckpt_dir",           type=str,   default=None)
    p.add_argument("--fig_dir",            type=str,   default=None)
    p.add_argument("--vis_every",          type=int,   default=10)
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag    = args.mode

    if args.ckpt_dir is None:
        args.ckpt_dir = str(_DIR / f"checkpoints_inv_{tag}")
    if args.fig_dir is None:
        args.fig_dir  = str(_DIR / f"figures_inv_{tag}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    ds_dir   = Path(args.dataset_dir)
    seg_len  = args.seg_len if args.seg_len > 0 else None
    train_ds = VTCMInverseDataset(ds_dir / f"train{args.train_suffix}.hdf5",
                                  dt=args.dt, seg_len=seg_len, augment=True)
    val_ds   = VTCMInverseDataset(ds_dir / f"validation{args.val_suffix}.hdf5",
                                  dt=args.dt, seg_len=seg_len, augment=False)
    dt = args.dt
    fs = 1.0 / dt

    print("=" * 80)
    print(f"  VTCM Inverse Problem — Physics-Guided Residual FNO  [mode={tag.upper()}]")
    print("=" * 80)
    print(f"  Direction   : y [1 Body-Z, T]  →  u [1 vertical irregularity, T]")
    print(f"  Device      : {device}")
    print(f"  Train / Val : {len(train_ds)} / {len(val_ds)} samples")
    seg_info = f"seg_len={seg_len}" if seg_len else "seg_len=full"
    print(f"  dt={dt*1e3:.2f} ms  fs={fs:.0f} Hz  {seg_info}  wd={args.weight_decay}  patience={args.early_stop_patience}")
    print("=" * 80)

    # ── FRF Identification ───────────────────────────────────────────────────
    use_residual = args.mode in ("residual", "both")
    use_frf_loss = args.mode in ("frf_loss", "both")
    need_frf     = use_residual or use_frf_loss

    identifier = None
    if need_frf:
        identifier = FRFIdentifier(dt=dt, n_avg=args.frf_n_avg,
                                   msc_threshold=args.msc_threshold)
        identifier.run(train_ds, n_samples=args.frf_samples,
                       train_split=args.frf_train_split)

    # ── Physics modules ──────────────────────────────────────────────────────
    green_inv   = None
    frf_loss_fn = None

    if use_residual and identifier is not None:
        green_inv = GreenInverseBaseline(
            H_emp=identifier.H_emp, msc_mask=identifier.msc_mask,
            dt=dt, reg=args.wiener_reg,
        ).to(device)
        print(f"  [Strategy A] GreenInverseBaseline  (Wiener reg={args.wiener_reg})")

    if use_frf_loss and identifier is not None:
        frf_loss_fn = ForwardFRFConsistencyLoss(
            H_emp=identifier.H_emp, msc_mask=identifier.msc_mask, dt=dt,
        ).to(device)
        print(f"  [Strategy B] ForwardFRFConsistencyLoss  "
              f"(active bins={int(identifier.msc_mask.sum())})")

    # ── Strategy C: Physics Forward Reconstruction Loss ───────────────────────
    # Always enabled when FRF is available (identifier has been run)
    # and lambda_physics > 0. Works for all modes since it only needs H_emp.
    physics_loss_fn = None
    if args.lambda_physics > 0 and identifier is not None:
        physics_loss_fn = PhysicsForwardLoss(
            H_emp=identifier.H_emp, dt=dt, win_len=args.phys_win_len,
        ).to(device)
        print(f"  [Strategy C] PhysicsForwardLoss  (time-domain FRF forward, "
              f"λ_phys={args.lambda_physics}, win={args.phys_win_len})")
    elif args.lambda_physics > 0 and identifier is None:
        # FRF not identified yet — run identification first then build loss
        identifier = FRFIdentifier(dt=dt, n_avg=args.frf_n_avg,
                                   msc_threshold=args.msc_threshold)
        identifier.run(train_ds, n_samples=args.frf_samples,
                       train_split=args.frf_train_split)
        physics_loss_fn = PhysicsForwardLoss(
            H_emp=identifier.H_emp, dt=dt, win_len=args.phys_win_len,
        ).to(device)
        print(f"  [Strategy C] PhysicsForwardLoss  (FRF identified on-demand, "
              f"λ_phys={args.lambda_physics}, win={args.phys_win_len})")

    # ── FNO backbone ─────────────────────────────────────────────────────────
    fno = FNO(
        in_channels        = N_DOF,            # 21 DOF vehicle → input
        out_channels       = N_RAIL,           # 2 rail channels → output
        decoder_layers     = args.decoder_layers,
        decoder_layer_size = args.decoder_layer_size,
        dimension          = 1,
        latent_channels    = args.latent_channels,
        num_fno_layers     = args.num_fno_layers,
        num_fno_modes      = args.num_fno_modes,
        padding            = args.padding,
    ).to(device)

    model = ResidualInverseFNO(fno=fno, green_inv=green_inv).to(device)
    n_params = sum(p.numel() for p in fno.parameters() if p.requires_grad)
    print(f"  FNO params  : {n_params:,}  (in={N_DOF} → out={N_RAIL})")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    # ── Optimiser / Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(fno.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.lr_T_max, eta_min=args.lr_eta_min)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    history = {"train": [], "val": [], "frf_train": [], "frf_val": [],
               "phys_train": [], "phys_val": []}

    best_val_loss    = float("inf")
    patience_counter = 0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        s_loss = s_data = s_frf = s_phys = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for batch in pbar:
            x      = batch["x"].to(device, non_blocking=True)
            target = batch["u"].to(device, non_blocking=True)

            optimizer.zero_grad()
            u_pred, _ = model(x)

            loss_data = F.mse_loss(u_pred, target)
            loss_frf  = torch.tensor(0.0, device=device)
            loss_phys = torch.tensor(0.0, device=device)

            if frf_loss_fn is not None:
                loss_frf, _ = frf_loss_fn(u_pred, x)
            if physics_loss_fn is not None:
                loss_phys, _ = physics_loss_fn(u_pred, x)

            loss = (loss_data
                    + args.lambda_frf     * loss_frf
                    + args.lambda_physics * loss_phys)
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(fno.parameters(), args.clip_grad)
            optimizer.step()

            s_loss += loss.item()
            s_data += loss_data.item()
            s_frf  += loss_frf.item()
            s_phys += loss_phys.item()
            pbar.set_postfix(
                loss=f"{loss.item():.3e}",
                data=f"{loss_data.item():.3e}",
                phys=f"{loss_phys.item():.3e}",
            )

        scheduler.step()
        nb = len(train_loader)
        history["train"].append(s_loss / nb)
        history["frf_train"].append(s_frf / nb)
        history["phys_train"].append(s_phys / nb)

        val_res = val_epoch(model, val_loader, frf_loss_fn, physics_loss_fn,
                            device, args.lambda_frf, args.lambda_physics,
                            epoch, args.fig_dir, args.vis_every)
        history["val"].append(val_res["loss"])
        history["frf_val"].append(val_res["frf"])
        history["phys_val"].append(val_res["phys"])

        r2_str = "  ".join(
            f"R²_ch{ch}={val_res.get(f'r2_norm_ch{ch}', float('nan')):.3f}"
            for ch in range(N_RAIL)
        )
        print(
            f"[Ep {epoch+1:4d}/{args.epochs}]"
            f"  train={s_loss/nb:.4e}  val={val_res['loss']:.4e}"
            f"  frf_tr={s_frf/nb:.3e}  frf_val={val_res['frf']:.3e}"
            f"  phys_tr={s_phys/nb:.3e}  phys_val={val_res['phys']:.3e}"
            f"  {r2_str}"
            f"  lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        # ── Early stopping & best model ────────────────────────────────────
        cur_val = val_res["loss"]
        if cur_val < best_val_loss:
            best_val_loss    = cur_val
            patience_counter = 0
            # save best model manually
            best_path = os.path.join(args.ckpt_dir, "best_model.pth")
            torch.save({"epoch": epoch, "model": fno.state_dict(),
                        "val_loss": best_val_loss}, best_path)
        else:
            patience_counter += 1

        save_checkpoint(args.ckpt_dir, models=fno,
                        optimizer=optimizer, scheduler=scheduler, epoch=epoch)

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"\n[Early Stop] val loss did not improve for "
                  f"{args.early_stop_patience} epochs. Best={best_val_loss:.4e}")
            break

    # ── Loss curves ──────────────────────────────────────────────────────────
    plt.rcParams.update(_PLOT_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ep = np.arange(1, args.epochs + 1)

    ax1.semilogy(ep, history["train"], color="#1B4F72", label="Train")
    ax1.semilogy(ep, history["val"],   color="#C0392B", ls="--", label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss")
    ax1.set_title("(a)  Total Loss"); ax1.legend()
    ax1.grid(True, which="both", ls=":", lw=0.5)

    ax2.semilogy(ep, np.array(history["frf_train"]) + 1e-15, color="#1E8449", label="Train FRF")
    ax2.semilogy(ep, np.array(history["frf_val"]) + 1e-15,   color="#D35400", ls="--", label="Val FRF")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("FRF Consistency Loss")
    ax2.set_title("(b)  Forward FRF Consistency"); ax2.legend()
    ax2.grid(True, which="both", ls=":", lw=0.5)

    fig.suptitle(f"Training History — mode={args.mode}  λ_frf={args.lambda_frf}",
                 fontsize=12, fontweight="bold")
    os.makedirs(args.fig_dir, exist_ok=True)
    fig.savefig(os.path.join(args.fig_dir, "loss_history.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nLoss history → {args.fig_dir}/loss_history.png")
    print(f"Checkpoints  → {args.ckpt_dir}/")


if __name__ == "__main__":
    main()
