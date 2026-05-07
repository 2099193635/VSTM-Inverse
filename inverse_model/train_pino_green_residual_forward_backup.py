"""
train_pino_green_residual.py
================================================================================
Physics-Guided Residual FNO  —  VTCM Forward Dynamics
  Rail irregularity u(t) → Vehicle 21-DOF response y(t)

Based on the Green's function feasibility analysis (test_green_function.py),
two complementary strategies are implemented and can be freely combined:

  Strategy A  ─ Residual Learning  (--mode residual | both)
    y_pred[dof] = GreenConv(u_excit[dof], g[dof]) + FNO_θ(u)[dof]
    ‣ Green's function convolution provides the linear baseline (corr ≈ 0.70);
      FNO only has to learn the non-linear residuals → faster convergence.

  Strategy B  ─ Band-Masked FRF Loss  (--mode frf_loss | both)
    L_frf = Σ_dof Σ_ω  M[dof,ω] · |Y_pred[dof,ω] − H[dof,ω]·U[ω]|²
    ‣ M[dof,ω] = 1  if  MSC[dof,ω] > msc_threshold  else  0
    ‣ Only enforce the FRF constraint in frequency bands where the system is
      confirmed to behave linearly (high coherence), avoiding over-constraining
      the strongly non-linear bands (Roll DOFs, contact impacts).

CLI --mode choices:
  baseline   ─ vanilla FNO (pure data-driven, no physics)
  residual   ─ Strategy A only
  frf_loss   ─ Strategy B only
  both       ─ Strategy A + B  [recommended]

Modal decoupling (from test_green_function.py insight):
  Z / Y DOFs  →  u_sym  = (u_L + u_R) / 2   (symmetric excitation)
  Roll DOFs   →  u_asym = u_L − u_R          (anti-symmetric excitation)

Dataset:  datasets/VTCM_vertical/{train,validation}_full_seq.hdf5
  input  : [N, 2, T]  ─ left/right rail absolute elevation (m)
  output : [N, 21, T] ─ vehicle 21-DOF absolute displacement (m)
  dt     : [N]        ─ time step per sample (s)
  vx_mps : [N]        ─ speed per sample (m/s)
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
DOF_LABELS = [
    "Body-Z",    "Body-Y",    "Body-Roll",
    "Bogie1-Z",  "Bogie1-Y",  "Bogie1-Roll",
    "Bogie2-Z",  "Bogie2-Y",  "Bogie2-Roll",
    "Axle1-Z",   "Axle1-Y",   "Axle1-Roll",
    "Axle2-Z",   "Axle2-Y",   "Axle2-Roll",
    "Axle3-Z",   "Axle3-Y",   "Axle3-Roll",
    "Axle4-Z",   "Axle4-Y",   "Axle4-Roll",
]
N_DOF = 21

def _is_roll(dof: int) -> bool:
    """Return True for Roll DOFs (index % 3 == 2)."""
    return (dof % 3 == 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class VTCMVerticalDataset(Dataset):
    """
    Loads VTCM_vertical HDF5 files.
    Returns bandpass-filtered, modally-decoupled tensors per sample.

    Input tensor layout  [3, T]:
        ch0 = u_sym   = (u_L + u_R) / 2  (vertical symmetric excitation)
        ch1 = u_asym  = u_L − u_R         (anti-symmetric / roll excitation)
        ch2 = u_L_raw bandpassed          (for reference / FRF loss scaling)

    Output tensor layout [21, T]:
        vehicle 21-DOF bandpass-filtered relative displacements
        y_rel[dof] = y_bp[dof] − u_L_bp  (removes common slope)
    """

    BP_LOW  = 0.5    # Hz
    BP_HIGH = 50.0   # Hz

    def __init__(self, h5_path: str | Path, return_raw: bool = False):
        self.h5_path   = Path(h5_path)
        self.return_raw = return_raw
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            self._input  = np.asarray(f["input"],   dtype=np.float32)  # [N, 2, T]
            self._output = np.asarray(f["output"],  dtype=np.float32)  # [N, 21, T]
            self._dt     = np.asarray(f["dt"],      dtype=np.float64)  # [N]
            self._vx     = np.asarray(f["vx_mps"],  dtype=np.float32)  # [N]

    def __len__(self) -> int:
        return self._input.shape[0]

    def _bandpass_sos(self, dt: float):
        fs  = 1.0 / dt
        nyq = fs / 2.0
        return sp_signal.butter(
            4,
            [self.BP_LOW / nyq, min(self.BP_HIGH / nyq, 0.99)],
            btype="band", output="sos",
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        dt     = float(self._dt[idx])
        sos    = self._bandpass_sos(dt)
        bp     = lambda x: sp_signal.sosfiltfilt(sos, x).astype(np.float32)

        u_L    = bp(self._input[idx, 0])          # [T]
        u_R    = bp(self._input[idx, 1])          # [T]
        u_sym  = (u_L + u_R) / 2.0               # symmetric (Z/Y excitation)
        u_asym = u_L - u_R                        # anti-symmetric (Roll excitation)

        # Relative displacement per DOF (bandpassed vehicle − left rail reference)
        y_rel  = np.stack([
            bp(self._output[idx, dof]) - (np.zeros_like(u_L) if _is_roll(dof) else u_L)
            for dof in range(N_DOF)
        ], axis=0)  # [21, T]

        # Input: [3, T]  (sym | asym | u_L_bp for bookkeeping)
        u_inp = np.stack([u_sym, u_asym, u_L], axis=0)  # [3, T]

        sample = {
            "input":  torch.from_numpy(u_inp),           # [3, T]
            "output": torch.from_numpy(y_rel),            # [21, T]
            "dt":     torch.tensor(dt, dtype=torch.float32),
            "vx":     torch.tensor(float(self._vx[idx]), dtype=torch.float32),
        }
        if self.return_raw:
            sample["input_raw"]  = torch.from_numpy(self._input[idx])   # [2, T]
            sample["output_raw"] = torch.from_numpy(self._output[idx])  # [21, T]
        return sample


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Pre-training FRF Identification
# ═══════════════════════════════════════════════════════════════════════════════

class FRFIdentifier:
    """
    Offline FRF identification using Welch cross-spectrum method.
    Runs once before training; produces:
      g_kernels : np.ndarray  [N_DOF, L_g]   – truncated impulse-response kernels
      H_emp     : np.ndarray  [N_DOF, F]     – averaged empirical FRF (complex)
      msc       : np.ndarray  [N_DOF, F]     – averaged MSC
      freqs     : np.ndarray  [F]            – frequency axis (Hz)
      dt        : float
    """

    def __init__(
        self,
        dt: float,
        n_avg: int = 8,
        msc_threshold: float = 0.6,
        g_min_pts: int = 512,
    ):
        self.dt            = dt
        self.n_avg         = n_avg
        self.msc_threshold = msc_threshold
        self.g_min_pts     = g_min_pts

    # ── Welch helpers ──────────────────────────────────────────────────────────
    def _welch_frf(self, u: np.ndarray, y: np.ndarray):
        T       = len(u)
        nperseg = max(512, T // self.n_avg)
        noverlap = nperseg // 2
        fs      = 1.0 / self.dt
        f, Suu = sp_signal.welch(u, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                 window="hann", return_onesided=True)
        _, Syu = sp_signal.csd(u, y, fs=fs, nperseg=nperseg, noverlap=noverlap,
                               window="hann", return_onesided=True)
        reg    = 1e-6 * np.max(np.abs(Suu))
        H      = Syu / (Suu + reg)
        return f, H

    def _welch_coh(self, u: np.ndarray, y: np.ndarray):
        T       = len(u)
        nperseg = max(512, T // self.n_avg)
        noverlap = nperseg // 2
        fs      = 1.0 / self.dt
        f, coh = sp_signal.coherence(u, y, fs=fs, nperseg=nperseg,
                                     noverlap=noverlap, window="hann")
        return f, coh

    def _frf_to_impulse(self, H: np.ndarray) -> np.ndarray:
        N      = (len(H) - 1) * 2
        H_full = np.zeros(N, dtype=complex)
        H_full[:len(H)] = H
        H_full[len(H):] = np.conj(H[-2:0:-1])
        g = np.real(np.fft.ifft(H_full)) / self.dt
        return g

    def _truncate_kernel(self, g: np.ndarray) -> np.ndarray:
        env     = np.abs(sp_signal.hilbert(g))
        peak    = env.max()
        cut_idx = np.argmax(env < peak * 0.001)
        if cut_idx == 0:
            cut_idx = len(g) // 4
        return g[:max(cut_idx, self.g_min_pts)]

    # ── Main identification routine ────────────────────────────────────────────
    def run(self, dataset: VTCMVerticalDataset, n_samples: Optional[int] = None,
            train_split: float = 0.6) -> None:
        """
        Average FRF across multiple training samples for robustness.
        Uses only the first `train_split` fraction of each sample as "training data"
        to avoid leakage when we later predict on the full sequence.
        """
        n_samples = min(n_samples or len(dataset), len(dataset))
        print(f"\n[FRF Identification]  samples={n_samples}, n_avg={self.n_avg}, "
              f"msc_threshold={self.msc_threshold}")

        # Accumulate across samples
        H_accum   = None   # [N_DOF, F] complex
        coh_accum = None   # [N_DOF, F]
        n_valid   = np.zeros(N_DOF, dtype=int)
        freqs_ref = None

        for idx in tqdm(range(n_samples), desc="  FRF", leave=False):
            item = dataset[idx]
            dt_s = float(item["dt"].item())

            u_sym  = item["input"][0].numpy()   # [T]
            u_asym = item["input"][1].numpy()   # [T]
            T      = len(u_sym)
            n_tr   = int(T * train_split)

            u_sym_tr  = u_sym[:n_tr]
            u_asym_tr = u_asym[:n_tr]

            for dof in range(N_DOF):
                y_tr = item["output"][dof].numpy()[:n_tr]
                u_tr = u_asym_tr if _is_roll(dof) else u_sym_tr
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f, H   = self._welch_frf(u_tr, y_tr)
                        _, coh = self._welch_coh(u_tr, y_tr)

                    if H_accum is None:
                        F_len     = len(f)
                        H_accum   = np.zeros((N_DOF, F_len), dtype=complex)
                        coh_accum = np.zeros((N_DOF, F_len), dtype=float)
                        freqs_ref = f

                    F_use = min(len(f), H_accum.shape[1])
                    H_accum[dof, :F_use]   += H[:F_use]
                    coh_accum[dof, :F_use] += coh[:F_use]
                    n_valid[dof] += 1
                except Exception:
                    pass

        # Average
        for dof in range(N_DOF):
            if n_valid[dof] > 0:
                H_accum[dof]   /= n_valid[dof]
                coh_accum[dof] /= n_valid[dof]

        self.freqs    = freqs_ref           # [F]
        self.H_emp    = H_accum             # [N_DOF, F] complex
        self.msc      = coh_accum           # [N_DOF, F]
        self.msc_mask = (coh_accum >= self.msc_threshold).astype(np.float32)  # [N_DOF, F]

        # Build impulse-response kernels
        g_list = []
        L_max  = 0
        for dof in range(N_DOF):
            g_full  = self._frf_to_impulse(H_accum[dof])
            g_trunc = self._truncate_kernel(g_full)
            g_list.append(g_trunc)
            L_max = max(L_max, len(g_trunc))

        # Pad all kernels to the same length
        g_pad = np.zeros((N_DOF, L_max), dtype=np.float32)
        for dof, gk in enumerate(g_list):
            g_pad[dof, :len(gk)] = gk.astype(np.float32)
        self.g_kernels = g_pad   # [N_DOF, L_g]

        # Summary statistics
        avg_msc = float(np.mean(coh_accum))
        n_good  = int(np.sum(self.msc_mask.sum(axis=1) > 10))   # DOFs with >10 active freq bins
        print(f"  Done. F_bins={len(freqs_ref)}, L_g={L_max}, "
              f"avg_MSC={avg_msc:.3f}, DOFs with mask_bins>10: {n_good}/{N_DOF}")
        for dof in range(N_DOF):
            active = int(self.msc_mask[dof].sum())
            print(f"    {DOF_LABELS[dof]:<14}  avg_MSC={coh_accum[dof].mean():.3f}  "
                  f"mask_bins={active}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Physics Modules (differentiable, GPU-ready)
# ═══════════════════════════════════════════════════════════════════════════════

class GreenConvBaseline(nn.Module):
    """
    Strategy A: Fixed (no-gradient) Green's function convolution baseline.
    For each DOF, convolves the appropriate modal-decoupled excitation
    with the pre-identified impulse-response kernel g[dof].

    y_green[b, dof, :] = IFFT( FFT(g[dof]) × FFT(u_excit[b]) ) × dt
    """

    def __init__(self, g_kernels: np.ndarray, dt: float):
        """
        g_kernels : np.ndarray [N_DOF, L_g] – precomputed kernels
        dt        : float                   – time step (s)
        """
        super().__init__()
        self.dt = dt
        # Register as buffer → automatically moves to GPU with .to(device)
        self.register_buffer(
            "g",
            torch.from_numpy(g_kernels.astype(np.float32)),   # [N_DOF, L_g]
        )

    @torch.no_grad()
    def forward(
        self,
        u_sym:  torch.Tensor,   # [B, T]  symmetric excitation
        u_asym: torch.Tensor,   # [B, T]  anti-symmetric excitation
    ) -> torch.Tensor:          # [B, N_DOF, T]
        B, T   = u_sym.shape
        L_g    = self.g.shape[1]
        N_fft  = T + L_g - 1

        # Pre-compute FFTs of excitations
        U_sym  = torch.fft.rfft(u_sym,  n=N_fft)   # [B, F]
        U_asym = torch.fft.rfft(u_asym, n=N_fft)   # [B, F]

        results = []
        for dof in range(N_DOF):
            U = U_asym if _is_roll(dof) else U_sym   # [B, F]
            G = torch.fft.rfft(self.g[dof], n=N_fft) # [F]
            y = torch.fft.irfft(G.unsqueeze(0) * U, n=N_fft)  # [B, N_fft]
            results.append(y[:, :T] * self.dt)        # [B, T]

        return torch.stack(results, dim=1)            # [B, N_DOF, T]


class BandMaskedFRFLoss(nn.Module):
    """
    Strategy B: Band-Masked FRF Consistency Loss.

    For each DOF and each frequency bin ω where MSC > threshold:
        L[dof, ω] = |FFT(y_pred)[dof, ω] − H_emp[dof, ω] × FFT(u_excit)[ω]|²

    The mask M[dof, ω] ∈ {0,1} prevents the loss from penalising the
    model in strongly non-linear frequency bands (low MSC).
    """

    def __init__(
        self,
        H_emp:    np.ndarray,   # [N_DOF, F] complex
        msc_mask: np.ndarray,   # [N_DOF, F] float32 in {0,1}
        freqs:    np.ndarray,   # [F] Hz
        dt:       float,
        f_max:    float = 50.0,
    ):
        super().__init__()
        self.dt  = dt
        F_len    = H_emp.shape[1]

        # Restrict to f_max
        fs = 1.0 / dt
        f_axis = np.arange(F_len) * (fs / (2 * (F_len - 1)))  # approximate
        f_mask = (f_axis <= f_max).astype(np.float32)

        # Register H as real + imag buffers (complex not always serialisable)
        self.register_buffer("H_real",  torch.from_numpy(H_emp.real.astype(np.float32)))  # [N_DOF, F]
        self.register_buffer("H_imag",  torch.from_numpy(H_emp.imag.astype(np.float32)))
        self.register_buffer("mask",    torch.from_numpy(msc_mask * f_mask.reshape(1, -1)))  # [N_DOF, F]

    @property
    def H(self) -> torch.Tensor:
        return torch.complex(self.H_real, self.H_imag)   # [N_DOF, F]

    def forward(
        self,
        y_pred:  torch.Tensor,  # [B, N_DOF, T]
        u_sym:   torch.Tensor,  # [B, T]
        u_asym:  torch.Tensor,  # [B, T]
    ) -> tuple[torch.Tensor, dict]:
        B, n_dof, T = y_pred.shape
        F_model = T // 2 + 1
        F_h     = self.H_real.shape[1]
        F_use   = min(F_model, F_h)

        Y_pred  = torch.fft.rfft(y_pred,  dim=-1)              # [B, N_DOF, F_model]
        U_sym   = torch.fft.rfft(u_sym,   dim=-1)              # [B, F_model]
        U_asym  = torch.fft.rfft(u_asym,  dim=-1)              # [B, F_model]

        H       = self.H[:, :F_use]                            # [N_DOF, F_use]
        mask    = self.mask[:, :F_use]                         # [N_DOF, F_use]

        total_loss  = torch.tensor(0.0, device=y_pred.device)
        active_dofs = 0

        per_dof_loss = {}
        for dof in range(n_dof):
            m_d = mask[dof]              # [F_use]
            if m_d.sum() < 1.0:
                per_dof_loss[dof] = 0.0
                continue

            U  = U_asym[:, :F_use] if _is_roll(dof) else U_sym[:, :F_use]  # [B, F_use]
            H_d = H[dof].unsqueeze(0)                                        # [1, F_use]

            Y_exp  = H_d * U                                  # [B, F_use]  – expected from FRF
            Y_p    = Y_pred[:, dof, :F_use]                   # [B, F_use]

            diff   = Y_p - Y_exp                              # [B, F_use]
            # MSC-weighted squared error, normalised by number of active bins
            err    = (diff.real**2 + diff.imag**2) * m_d.unsqueeze(0)
            dof_loss = err.mean()
            total_loss  = total_loss + dof_loss
            active_dofs += 1
            per_dof_loss[dof] = float(dof_loss.item())

        if active_dofs > 0:
            total_loss = total_loss / active_dofs

        return total_loss, per_dof_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Residual FNO Model
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualFNO(nn.Module):
    """
    Strategy A wrapper around a standard FNO.

    Architecture:
        y_pred = GreenConvBaseline(u_sym, u_asym)  +  FNO(u_inp)
               ↑ fixed linear baseline               ↑ learns non-linear residual

    When use_green=False, reduces to a vanilla FNO (mode='baseline').
    """

    def __init__(
        self,
        fno:        FNO,
        green_conv: Optional[GreenConvBaseline] = None,
    ):
        super().__init__()
        self.fno        = fno
        self.green_conv = green_conv      # None → pure FNO
        self.use_green  = (green_conv is not None)

    def forward(self, u_inp: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        u_inp : [B, 3, T]  – ch0=u_sym, ch1=u_asym, ch2=u_L_ref

        Returns
        -------
        y_pred   : [B, 21, T]
        y_green  : [B, 21, T] or None
        """
        u_sym  = u_inp[:, 0, :]   # [B, T]
        u_asym = u_inp[:, 1, :]   # [B, T]

        y_fno = self.fno(u_inp)   # [B, 21, T]

        if self.use_green:
            y_green = self.green_conv(u_sym, u_asym)   # [B, 21, T]  (no grad)
            y_pred  = y_green + y_fno
        else:
            y_green = None
            y_pred  = y_fno

        return y_pred, y_green


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Training / Validation helpers
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


def _save_val_fig(
    pred:   np.ndarray,    # [21, T]
    target: np.ndarray,    # [21, T]
    epoch:  int,
    fig_dir: str,
    n_show:  int = 3,
) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update(_PLOT_RC)
    T      = pred.shape[1]
    t_axis = np.arange(T)
    dofs   = [0, 1, 9][:n_show]   # Body-Z, Body-Y, Axle1-Z

    fig, axes = plt.subplots(n_show, 2, figsize=(14, 3 * n_show), constrained_layout=True)
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, dof in enumerate(dofs):
        sig_t = target[dof]
        sig_p = pred[dof]

        ax = axes[row, 0]
        ax.plot(t_axis, sig_t, color="#1B4F72", lw=0.9, alpha=0.85, label="Ground truth")
        ax.plot(t_axis, sig_p, color="#C0392B", lw=0.9, ls="--", alpha=0.90, label="Prediction")
        corr = float(np.corrcoef(sig_p, sig_t)[0, 1]) if T > 1 else 0.0
        ax.set_title(f"{DOF_LABELS[dof]}  ($\\rho$={corr:.3f})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Displacement (m)")
        if row == 0:
            ax.legend(loc="upper right")

        from scipy.signal import welch
        nperseg = min(256, T // 4)
        f_t, psd_t = welch(sig_t, nperseg=nperseg)
        f_p, psd_p = welch(sig_p, nperseg=nperseg)
        ax2 = axes[row, 1]
        ax2.semilogy(f_t, psd_t, color="#1B4F72", lw=1.0, alpha=0.85, label="GT")
        ax2.semilogy(f_p, psd_p, color="#C0392B", lw=1.0, ls="--", alpha=0.90, label="Pred")
        ax2.set_title(f"PSD — {DOF_LABELS[dof]}")
        ax2.set_xlabel("Norm. freq.")
        ax2.set_ylabel("PSD")
        if row == 0:
            ax2.legend(loc="upper right")

    fig.suptitle(f"Validation @ Epoch {epoch + 1}", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(fig_dir, f"val_ep{epoch:04d}.png"), bbox_inches="tight")
    plt.close(fig)


def _compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Batch-averaged RMSE, normalised-R², per-DOF mean correlation."""
    pred_np   = pred.detach().cpu().numpy()     # [B, 21, T]
    target_np = target.detach().cpu().numpy()
    B, D, T   = pred_np.shape

    r2_list, corr_list = [], []
    for b in range(B):
        for dof in range(D):
            p, t = pred_np[b, dof], target_np[b, dof]
            rms_p = float(np.sqrt(np.mean(p**2))) + 1e-12
            rms_t = float(np.sqrt(np.mean(t**2))) + 1e-12
            p_sc  = p * (rms_t / rms_p)
            var_t = float(np.var(t)) + 1e-20
            r2_list.append(1.0 - float(np.mean((p_sc - t)**2)) / var_t)
            corr_list.append(float(np.corrcoef(p, t)[0, 1]))

    rmse = float(torch.sqrt(F.mse_loss(pred, target)).item())
    return {
        "rmse":    rmse,
        "r2_norm": float(np.nanmean(r2_list)),
        "corr":    float(np.nanmean(corr_list)),
    }


def val_epoch(
    model:       ResidualFNO,
    loader:      DataLoader,
    frf_loss_fn: Optional[BandMaskedFRFLoss],
    device:      torch.device,
    lambda_frf:  float,
    epoch:       int,
    fig_dir:     str,
    vis_every:   int,
) -> dict:
    model.eval()
    sum_loss = sum_data = sum_frf = 0.0
    n_batch  = 0
    viz_pred = viz_target = None

    with torch.no_grad():
        for batch in loader:
            u_inp  = batch["input"].to(device)    # [B, 3, T]
            target = batch["output"].to(device)   # [B, 21, T]
            y_pred, _ = model(u_inp)

            loss_data = F.mse_loss(y_pred, target)
            loss_frf  = torch.tensor(0.0, device=device)

            if frf_loss_fn is not None:
                u_sym  = u_inp[:, 0, :]
                u_asym = u_inp[:, 1, :]
                loss_frf, _ = frf_loss_fn(y_pred, u_sym, u_asym)

            loss = loss_data + lambda_frf * loss_frf
            sum_loss += loss.item()
            sum_data += loss_data.item()
            sum_frf  += loss_frf.item()
            n_batch  += 1

            if viz_pred is None:
                viz_pred   = y_pred[0].cpu().numpy()
                viz_target = target[0].cpu().numpy()

    n_batch = max(n_batch, 1)
    result  = dict(
        loss = sum_loss / n_batch,
        data = sum_data / n_batch,
        frf  = sum_frf  / n_batch,
    )
    if viz_pred is not None and (epoch % vis_every == 0):
        _save_val_fig(viz_pred, viz_target, epoch, fig_dir)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Physics-Guided Residual FNO for VTCM Dynamics (Green's function + Band-Masked FRF Loss)"
    )
    # ── Dataset ──────────────────────────────────────────────────────────────
    p.add_argument("--dataset_dir", type=str,
                   default=str(_ROOT / "datasets" / "VTCM_vertical"))
    p.add_argument("--train_suffix", type=str, default="_full_seq",
                   help="HDF5 suffix for training file (e.g. '' or '_full_seq')")
    p.add_argument("--val_suffix",   type=str, default="_full_seq")
    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--latent_channels",    type=int,   default=64)
    p.add_argument("--num_fno_layers",     type=int,   default=4)
    p.add_argument("--num_fno_modes",      type=int,   default=16)
    p.add_argument("--decoder_layers",     type=int,   default=2)
    p.add_argument("--decoder_layer_size", type=int,   default=64)
    p.add_argument("--padding",            type=int,   default=9)
    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--lr_T_max",    type=int,   default=200,
                   help="CosineAnnealingLR T_max (epochs)")
    p.add_argument("--lr_eta_min",  type=float, default=1e-5)
    p.add_argument("--clip_grad",   type=float, default=1.0,
                   help="Gradient clipping max-norm (0 = disabled)")
    # ── Physics mode ──────────────────────────────────────────────────────────
    p.add_argument("--mode", type=str, default="both",
                   choices=["baseline", "residual", "frf_loss", "both"],
                   help=("baseline=vanilla FNO | residual=Strategy A | "
                         "frf_loss=Strategy B | both=A+B"))
    p.add_argument("--lambda_frf",      type=float, default=0.05,
                   help="Weight of the band-masked FRF loss term")
    p.add_argument("--msc_threshold",   type=float, default=0.60,
                   help="MSC threshold above which the FRF constraint is enforced")
    p.add_argument("--frf_n_avg",       type=int,   default=8,
                   help="Welch segments for FRF identification")
    p.add_argument("--frf_samples",     type=int,   default=None,
                   help="Max training samples used for FRF identification (None=all)")
    p.add_argument("--frf_train_split", type=float, default=0.6,
                   help="Fraction of each sample used for FRF identification")
    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument("--ckpt_dir",  type=str, default=None)
    p.add_argument("--fig_dir",   type=str, default=None)
    p.add_argument("--vis_every", type=int, default=10)
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Auto-name output directories ─────────────────────────────────────────
    tag = args.mode
    if args.ckpt_dir is None:
        args.ckpt_dir = str(_DIR / f"checkpoints_green_residual_{tag}")
    if args.fig_dir is None:
        args.fig_dir  = str(_DIR / f"figures_green_residual_{tag}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    ds_dir    = Path(args.dataset_dir)
    train_h5  = ds_dir / f"train{args.train_suffix}.hdf5"
    val_h5    = ds_dir / f"validation{args.val_suffix}.hdf5"
    train_ds  = VTCMVerticalDataset(train_h5)
    val_ds    = VTCMVerticalDataset(val_h5)

    # Use dt from the first training sample (assumed homogeneous)
    dt = float(train_ds[0]["dt"].item())
    fs = 1.0 / dt

    print("=" * 80)
    print(f"  VTCM Physics-Guided Residual FNO  —  mode = {args.mode.upper()}")
    print("=" * 80)
    print(f"  Device     : {device}")
    print(f"  Train / Val: {len(train_ds)} / {len(val_ds)} samples")
    print(f"  dt         : {dt*1e3:.2f} ms  fs = {fs:.0f} Hz")
    print(f"  Train HDF5 : {train_h5}")
    print(f"  Val   HDF5 : {val_h5}")
    print(f"  lambda_frf : {args.lambda_frf}  msc_threshold : {args.msc_threshold}")
    print("=" * 80)

    # ── Step 1: Pre-training FRF Identification ───────────────────────────────
    use_residual = args.mode in ("residual", "both")
    use_frf_loss = args.mode in ("frf_loss", "both")
    need_frf     = use_residual or use_frf_loss

    identifier = None
    if need_frf:
        identifier = FRFIdentifier(
            dt=dt,
            n_avg=args.frf_n_avg,
            msc_threshold=args.msc_threshold,
        )
        identifier.run(
            train_ds,
            n_samples=args.frf_samples,
            train_split=args.frf_train_split,
        )

    # ── Step 2: Build physics modules ────────────────────────────────────────
    green_conv  = None
    frf_loss_fn = None

    if use_residual and identifier is not None:
        green_conv = GreenConvBaseline(identifier.g_kernels, dt).to(device)
        print(f"  [Strategy A] GreenConvBaseline: g_kernels shape = "
              f"{identifier.g_kernels.shape}")

    if use_frf_loss and identifier is not None:
        frf_loss_fn = BandMaskedFRFLoss(
            H_emp         = identifier.H_emp,
            msc_mask      = identifier.msc_mask,
            freqs         = identifier.freqs,
            dt            = dt,
            f_max         = VTCMVerticalDataset.BP_HIGH,
        ).to(device)
        active_total = int(identifier.msc_mask.sum())
        print(f"  [Strategy B] BandMaskedFRFLoss: total active freq bins = {active_total}")

    # ── Step 3: Build FNO ────────────────────────────────────────────────────
    fno_backbone = FNO(
        in_channels        = 3,           # u_sym | u_asym | u_L_ref
        out_channels       = N_DOF,       # 21 DOFs
        decoder_layers     = args.decoder_layers,
        decoder_layer_size = args.decoder_layer_size,
        dimension          = 1,
        latent_channels    = args.latent_channels,
        num_fno_layers     = args.num_fno_layers,
        num_fno_modes      = args.num_fno_modes,
        padding            = args.padding,
    ).to(device)

    model = ResidualFNO(fno=fno_backbone, green_conv=green_conv).to(device)

    n_params = sum(p.numel() for p in model.fno.parameters() if p.requires_grad)
    print(f"  FNO parameters: {n_params:,}")

    # ── Step 4: DataLoaders ───────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    ) if len(val_ds) > 0 else None

    # ── Step 5: Optimiser & Scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.fno.parameters(),   # Only FNO parameters; Green's conv has no grad
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = args.lr_T_max,
        eta_min = args.lr_eta_min,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Loss history for plotting
    history = {"train_loss": [], "val_loss": [], "train_frf": [], "val_frf": []}

    # ── Step 6: Training loop ─────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        sum_loss = sum_data = sum_frf = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for batch in pbar:
            u_inp  = batch["input"].to(device, non_blocking=True)    # [B, 3, T]
            target = batch["output"].to(device, non_blocking=True)   # [B, 21, T]

            optimizer.zero_grad()

            y_pred, y_green = model(u_inp)

            # ── Data loss ────────────────────────────────────────────────────
            loss_data = F.mse_loss(y_pred, target)

            # ── Band-Masked FRF loss (Strategy B) ────────────────────────────
            loss_frf = torch.tensor(0.0, device=device)
            if frf_loss_fn is not None:
                u_sym  = u_inp[:, 0, :]
                u_asym = u_inp[:, 1, :]
                loss_frf, _ = frf_loss_fn(y_pred, u_sym, u_asym)

            loss = loss_data + args.lambda_frf * loss_frf

            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.fno.parameters(), args.clip_grad)
            optimizer.step()

            sum_loss += loss.item()
            sum_data += loss_data.item()
            sum_frf  += loss_frf.item()

            pbar.set_postfix(
                loss=f"{loss.item():.3e}",
                data=f"{loss_data.item():.3e}",
                frf =f"{loss_frf.item():.3e}",
            )

        scheduler.step()
        n_b = len(train_loader)
        history["train_loss"].append(sum_loss / n_b)
        history["train_frf"].append(sum_frf / n_b)

        # ── Validation ───────────────────────────────────────────────────────
        val_result = {"loss": 0.0, "frf": 0.0}
        if val_loader is not None:
            val_result = val_epoch(
                model, val_loader, frf_loss_fn, device,
                args.lambda_frf, epoch, args.fig_dir, args.vis_every,
            )
        history["val_loss"].append(val_result["loss"])
        history["val_frf"].append(val_result["frf"])

        print(
            f"[Ep {epoch+1:4d}/{args.epochs}]"
            f"  train={sum_loss/n_b:.4e}  data={sum_data/n_b:.4e}"
            f"  frf_train={sum_frf/n_b:.4e}"
            f"  val={val_result['loss']:.4e}"
            f"  lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        save_checkpoint(
            args.ckpt_dir, models=model.fno,
            optimizer=optimizer, scheduler=scheduler, epoch=epoch,
        )

    # ── Step 7: Loss curve ────────────────────────────────────────────────────
    plt.rcParams.update(_PLOT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ep = np.arange(1, args.epochs + 1)
    axes[0].semilogy(ep, history["train_loss"], label="Train total", color="#1B4F72")
    axes[0].semilogy(ep, history["val_loss"],   label="Val total",   color="#C0392B", ls="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("(a)  Total Loss Curve")
    axes[0].legend()
    axes[0].grid(True, which="both", ls=":", lw=0.5)

    axes[1].semilogy(ep, np.array(history["train_frf"]) + 1e-12,
                     label="Train FRF loss", color="#1E8449")
    axes[1].semilogy(ep, np.array(history["val_frf"]) + 1e-12,
                     label="Val FRF loss",   color="#D35400", ls="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Band-Masked FRF Loss")
    axes[1].set_title("(b)  FRF Constraint Loss Curve")
    axes[1].legend()
    axes[1].grid(True, which="both", ls=":", lw=0.5)

    fig.suptitle(
        f"Training History  —  mode={args.mode}  λ_frf={args.lambda_frf}",
        fontsize=12, fontweight="bold",
    )
    os.makedirs(args.fig_dir, exist_ok=True)
    fig.savefig(os.path.join(args.fig_dir, "loss_history.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nLoss history saved → {args.fig_dir}/loss_history.png")
    print(f"Checkpoints saved  → {args.ckpt_dir}/")


if __name__ == "__main__":
    main()
