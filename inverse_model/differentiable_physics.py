'''
Author: 2099193635 2099193635@qq.com
Date: 2026-04-10 09:14:59
LastEditors: 2099193635 2099193635@qq.com
LastEditTime: 2026-04-10 16:07:26
FilePath: /VTCM_PYTHON/inverse_model/differentiable_physics.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import math

class ModalFRF(nn.Module):
    """
    Linearized Modal Frequency Response Function (FRF) Layer:
        Track spatial irregularity  → Vehicle body acceleration
    Physical model: Linearization of a Half-Car Model with n_modal Degrees of Freedom
    Acceleratrion FRF:
        H(ω) = Σᵢ  -ω² · Rᵢ / (ωᵢ² - ω² + 2i·ζᵢ·ωᵢ·ω)
    where:
        ωᵢ    : Natural Circular Frequency of the ith mode [rad/s]      ← Derived from the FiLM conditioning vector
        ζᵢ    : Damping Ratio of the ith mode (dimensionless)            ← Derived from the FiLM conditioning vector
        Rᵢ    : Modal Residue of the ith mode (m/s²·m⁻¹)  ← Learnable parameter
                     = φ_carbody,i · φ_contact,i    (the product of the mode shape at the carbody and the mode shape at the track contact point)

    Modal Residue Initialization Strategy:
        Centered on the theoretical value of a typical high-speed rail half-vehicle model.
        Fine-tuning during training is permitted.
        Constrained within the interval [-R_max, R_max] via the tanh function.
    
    Args:
        n_modal:        Modal oders
        n_sensors:      Number of sensors
        R_max:          Upper bound of the modal residue
        learnable:      if FRF modal residues are learnable
    """

    def __init__(
        self,
        n_modal: int = 8,
        n_sensors: int = 3,
        R_max: float = 50.0,
        learnable: bool = True,
    ) -> None:

        super().__init__()
        self.n_modal = n_modal
        self.n_sensors = n_sensors
        self.R_max = R_max

        init_R = torch.zeros(n_sensors, n_modal)
        for i in range(n_modal):
            scale = 1.0 / (1.0 + i) ** 0.5
            init_R[:, i] = scale * torch.ones(n_sensors)
        
        raw_R = torch.atanh(init_R / R_max).clamp(-3.0, 3.0)
        if learnable:
            self.raw_R = nn.Parameter(raw_R)
        else:
            self.register_buffer("raw_R", raw_R)
    
    @property
    def modal_residues(self) -> Tensor:
        return torch.tanh(self.raw_R) * self.R_max

    def forward(
            self,
            Z_kappa: Tensor,  # [B, F] Complex Frequency Spectrum
            omega_rad: Tensor,  # [B, F] Angular Frequency in rad/s
            nat_freqs: Tensor,  # [n_modal] Natural Circular Frequencies of the modes in rad/s
            damp_ratios: Tensor,  # [n_modal] Damping Ratios of the modes (dimensionless)
    ) -> Tensor:                # [B, F, n_sensors] Complex acceleration spectrum 
        """Calculate H(ω)·Z_r(ω) = A_c(ω)"""
        # ================ Demansion expansion ================
        # [B, F, 1]
        omega_q  = omega_rad.unsqueeze(-1).to(torch.complex64)
        # [B, 1, n_modal]
        omega_i  = nat_freqs.unsqueeze(1).to(torch.complex64)
        zeta_i   = damp_ratios.unsqueeze(1).to(torch.complex64)
        # [1, n_sensors, n_modal]
        R_i      = self.modal_residues.unsqueeze(0)

        # ================ Denominator of the Modal Residue  ===============
        # ωᵢ² - ω² + 2i·ζᵢ·ωᵢ·ω
        # shape broadcast → [B, F, n_modal]
        denom = (
            omega_i **2
            - omega_q **2
            + 2j * zeta_i * omega_i * omega_q
        )

        # avoid nan by adding a small epsilon to the denominator
        denom = torch.where(
            denom.abs() < 1e-8,
            torch.full_like(denom, 1e-8),
            denom,
        )

        # ================ Modal Acceleration FRF Calculation ================
        # H(ω) = -ω² · Σᵢ Rᵢ/denomᵢ
        # R_i: [1, n_s, n_modal], denom: [B, F, n_modal] → need broadcast
        # Reshape denom to [B, F, 1, n_modal] for broadcasting
        denom_4d = denom.unsqueeze(2)                        #  [B, F, 1, n_modal]
        R_4d = R_i.unsqueeze(1).to(torch.complex64)          #  [1, 1, n_s, n_modal]
        # H: [B, F, n_sensors]
        H = (-(omega_q ** 2) * (R_4d / denom_4d).sum(-1))

        # ================ Output Calculation ================
        # A_c(ω) = H(ω)·Z_r(ω)
        # Z_kappa: [B, F] → [B, F, 1]
        A_omega = H * Z_kappa.unsqueeze(-1).to(torch.complex64)  # [B, F, n_sensors]
        return A_omega


class VehicleTrackFRFPhysics(nn.Module):
    """
    Complete differentiable FRF physical layer: 
        Track spatial irregularity → vehicle body time domain accleration
    
    Process:
        1. z_pred [B, L]   → FFT → Z_kappa [B, F]    (frequency domain transformation)
        2. κ → ω             ω = 2π · κ · v_x        (coordinate transformation)
        3. ModalFRF          Z_kappa · H(ω, nat_freqs, damp_ratios) → A_omega [B, F, n_s]
        4. IFFT              A_omega → a_c [B, T, n_s] (time domain transformation) 

    Key: L(Number of Spatial Sampling Points) can be different from T(Number of Temporal Sampling Points), which is more flexible for the training of Inverse model. 
        T = round(L · dx / dt / v_x) = round(D / (v_x · dt))
        The output length is controlled via the irfft(n=T).
    
    Args:
        n_modal:        Modal oders
        n_sensors:      Number of sensors
        dx:             Spatial sampling interval [m]
        dt:             Temporal sampling interval [s]
        R_max:          Upper bound of the modal residue
        frf_learnable:  if FRF modal residues are learnable
    """
    def __init__(
        self,
        n_modal:    int = 8,
        n_sensors:  int = 3,
        dx:         float = 0.25,
        dt:         float = 1e-4,
        R_max:      float = 50.0,
        learnable:  bool = True,
            ) -> None:
        super().__init__()
        self.n_modal = n_modal
        self.n_sensors = n_sensors
        self.dx = dx
        self.dt = dt
        self.frf = ModalFRF(
            n_modal=n_modal,
            n_sensors=n_sensors,
            R_max=R_max,
            learnable=learnable,
        )

    def _compute_time_steps(self, L: int, v_x: Tensor) -> int:
        """Compute the number of time steps T based on the input length L and velocity v_x"""
        D = L * self.dx                              # Total spatial length
        T_sec = D / v_x                               # Number of time steps in seconds
        return max(1, round(T_sec / self.dt))
    
    def forward(
        self,
        z_pred: Tensor,          # [B, L] Track irregularity in spatial domain
        vx_mps: Tensor,             # [B] Velocity in m/s
        nat_freqs: Tensor,       # [B, n_modal] Natural Circular Frequencies of the modes in rad/s
        damp_ratios: Tensor,     # [B, n_modal] Damping Ratios of the modes (dimensionless)
        T_target: Optional[int] = None,  # Target number of time steps for the output (if None, it will be computed based on L and v_x)
    ) -> Tensor:                 # [B, T, n_sensors] Vehicle body acceleration in time domain
        """
        Mapping from track irregularity to vehicle body acceleration
        """
        if z_pred.dim() == 3:
            z_pred = z_pred.squeeze(-1)    # [B, L]
        B, L = z_pred.shape

        # ============== Spatial FFT ==============
        Z_kappa = torch.fft.rfft(z_pred, dim = -1)    # [B, F], F = L//2 + 1
        kappa = torch.fft.rfftfreq(L, d=self.dx, device = z_pred.device)  # [F]

        # ============== Coordinate Transformation: κ → ω = 2π · κ · v_x ==============
        # kappa: [F], vx_mps: [B] → omega: [B, F]
        omega_rad = 2.0 * math.pi * kappa.unsqueeze(0) * vx_mps.unsqueeze(1)

        # Step 3: Modal FRF Calculation
        A_omega = self.frf(Z_kappa, omega_rad, nat_freqs, damp_ratios)  # [B, F, n_sensors]

        # Step 4: Frequency to Time Domain Transformation
        if T_target is None:
            vx_mean = vx_mps.mean().item()
            T_target = self._compute_time_steps(L, vx_mean)
        
        # irfft based on demision F → [B, T, n_s]
        a_c_pred = torch.fft.irfft(
            A_omega.permute(0, 2, 1),  # [B, n_sensors, F]
            n=T_target, 
            dim=-1
            )    # [B, n_sensors, T]
        return a_c_pred.transpose(1, 2).contiguous()  # [B, T, n_sensors]

# PINN residial
class TrackODEPINN(nn.Module):
    """
    PINN physical layer for track ODE residuals calculation

    Simplified Euler-Bernoulli Rail Beam Equation:
        EI · z⁴ + k_b · z = -F_contact(z)
    
    where:
        w: Rail deflection
        EI: Bending stiffness of the rail
        k_b: Track stiffness
        k_Hz: Damping coefficient
    F_contact (Linearized Hertz):
        F_contact ≈ k_Hz · (z_wheel_static - z)  for z < z_eq
                  ≈ 0                             for z ≥ z_separation
    Using Spectral Differentiation to Compute High-Order Spatial Derivatives

    Args:
        EI:      Bending stiffness of the rail
        k_b:     Track stiffness
        k_Hz:    Damping coefficient
        dx:      Spatial sampling interval [m]
    """

    def __init__(
        self,
        EI:   float = 6.4e6,
        k_b:  float = 5.0e7,
        k_Hz: float = 1.2e9,
        dx:   float = 0.25,
    ) -> None:
        super().__init__()
        self.register_buffer("EI",   torch.tensor(EI))
        self.register_buffer("k_b",  torch.tensor(k_b))
        self.register_buffer("k_Hz", torch.tensor(k_Hz))
        self.dx = dx
    
    def _spectral_diff4(self, z: Tensor) -> Tensor:
        """
        Compute the 4th order spatial derivative using spectral differentiation
        z: [B, L] → z⁴: [B, L]

        Args:
            z: [B, L] Rail deflection in spatial domain
        Returns:
            z⁴: [B, L] Fourth-order spatial derivative of the rail deflection
        """

        B, L = z.shape
        Z = torch.fft.rfft(z, dim=-1)                               # [B, F], F = L//2 + 1
        kappa = torch.fft.rfftfreq(L, d=self.dx, device=z.device)   # [F]
        omega_k = (2.0 * math.pi * kappa) ** 4                      # [F]
        d4Z = Z * omega_k.unsqueeze(0)                              # [B, F]
        return torch.fft.irfft(d4Z, n=L, dim=-1)                    # [B, L]

    def forward(self, z_pred: Tensor) -> Tensor:
        """
        Compute the ODE residuals based on the predicted rail deflection z_pred
        The smaller the residuals, the better the predicted deflection satisfies the physical ODE.
        Args:
            z_pred: [B, L] Predicted rail deflection in spatial domain
        Returns:
            residuals: [B, L] ODE residuals at each spatial point
        """
        if z_pred.dim() == 3:
            z_pred = z_pred.squeeze(-1)    # [B, L]
        
        # EI · z'''' (spectral differentiation) 
        d4z = self._spectral_diff4(z_pred)  # [B, L]

        # Linearized contact force
        # F_contact ≈ k_Hz · z  assuming z_wheel_static = 0 and linearization around the static equilibrium position
        F_contact = self.k_Hz * z_pred  # [B, L]

        # ODE Residuals: EI · z⁴ + k_b · z + F_contact
        residuals = self.EI * d4z + self.k_b * z_pred + F_contact  # [B, L]

        # Normalize residuals to stabilize training
        residual_normalized = residuals / (self.EI.item() + 1.0)
        return (residual_normalized ** 2).mean()

class PhysicsLayer(nn.Module):
    """
    Unified Physical Layer Scheduling Interface

    It supports three modes:
      1. frf : FRF frequency domain embedding
      2. pinn: PINN track ODE residuals
      3. both: FRF + PINN

      Args:
        mode:           Physical mode, "frf" / "pinn" / "both"
        n_modal:        FRF modal orders
        n_sensors:      Number of sensors (for FRF)
        dx:             Spatial sampling interval [m]
        dt:             Temporal sampling interval [s]
        EI, k_b, k_Hz:  Track parameters for PINN
        frf_learnable:  if FRF modal residues are learnable
    """

    def __init__(self, 
                 mode:      str      = "frf",
                 n_modal:   int      = 8,
                 n_sensors: int      = 3,
                 dx:        float    = 0.25,
                 dt:        float    = 1e-4,
                 EI:        float    = 6.4e6,
                 k_b:       float    = 5.0e7,
                 k_Hz:      float    = 1.2e9,
                 frf_learnable: bool = True, 
                 ) -> None:
        super().__init__()
        self.mode = mode
        if mode in ("frf", "both"):
            self.frf_layer = VehicleTrackFRFPhysics(
                n_modal=n_modal,
                n_sensors=n_sensors,
                dx=dx,
                dt=dt,
                R_max=50.0,
                learnable=frf_learnable,
            )

        if mode in ("pinn", "both"):
            self.pinn_layer = TrackODEPINN(
                EI=EI,
                k_b=k_b,
                k_Hz=k_Hz,
                dx=dx,
            )
        
    def compute_frf_loss(
        self,
        z_pred: Tensor,             # [B, L] Track irregularity in spatial domain
        vx_mps: Tensor,             # [B] Velocity in m/s
        nat_freqs: Tensor,          # [B, n_modal] Natural Circular Frequencies of the modes in rad/s
        damp_ratios: Tensor,        # [B, n_modal] Damping Ratios of the modes (dimensionless)
        a_c_true: Tensor,           # [B, T, n_sensors] True vehicle body acceleration in time domain
    ) -> Tensor:
        """FRF LOSS: Use L1 loss"""
        T_target = a_c_true.shape[1]
        a_c_pred = self.frf_layer(
            z_pred, vx_mps, nat_freqs, damp_ratios, T_target=T_target
        )  # [B, T, n_sensors]
        return torch.nn.functional.l1_loss(a_c_pred, a_c_true)

    def compute_pinn_loss(
        self,
        z_pred: Tensor,             # [B, L] Track irregularity in spatial domain
    ) -> Tensor:
        """PINN LOSS: Use mean squared ODE residuals"""
        return self.pinn_layer(z_pred)
    
    def forward(
        self,
        z_pred: Tensor,             # [B, L] Track irregularity in spatial domain
        a_c_true: Optional[Tensor] = None,           # [B, T, n_sensors] True vehicle body acceleration in time domain (required for FRF)
        nat_freqs: Optional[Tensor] = None,          # [B, n_modal] Natural Circular Frequencies of the modes in rad/s (required for FRF)
        damp_ratios: Optional[Tensor] = None,        # [B, n_modal] Damping Ratios of the modes (dimensionless) (required for FRF)
        vx_mps: Optional[Tensor] = None,             # [B] Velocity in m/s (required for FRF)
        lambda_frf: float = 1.0,                     # Weight for FRF loss
        lambda_pinn: float = 1.0,                    # Weight for PINN loss
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the combined physics-informed loss based on the selected mode (FRF, PINN, or both)
        Returns:
            loss: scalar physical loss
            metrics: {"l_frf": ..., "l_pinn": ...}
        """
        l_frf  = torch.tensor(0.0, device=z_pred.device)
        l_pinn = torch.tensor(0.0, device=z_pred.device)

        if self.mode in ("frf", "both"):
            assert a_c_true is not None, "a_c_true required for FRF mode"
            l_frf = self.compute_frf_loss(
                z_pred, vx_mps, nat_freqs, damp_ratios, a_c_true
            )

        if self.mode in ("pinn", "both"):
            l_pinn = self.compute_pinn_loss(z_pred)

        loss = lambda_frf * l_frf + lambda_pinn * l_pinn
        return loss, {"l_frf": l_frf.item(), "l_pinn": l_pinn.item()}