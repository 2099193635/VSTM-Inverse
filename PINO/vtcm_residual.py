from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class ResidualSummary:
    dynamics_residual: torch.Tensor
    displacement_residual: torch.Tensor
    velocity_residual: torch.Tensor
    loss_dynamics: torch.Tensor
    loss_kinematics: torch.Tensor
    loss_total: torch.Tensor


class VTCMDynamicsResidual(nn.Module):
    def __init__(
        self,
        mass_vector: torch.Tensor,
        dt: float,
        alpha: float = 0.5,
        beta: float = 0.25,
        dynamics_weight: float = 1.0,
        kinematic_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if mass_vector.ndim != 1:
            raise ValueError("mass_vector 必须是一维张量。")
        if dt <= 0.0:
            raise ValueError("dt 必须大于 0。")

        self.register_buffer("mass_vector", mass_vector.view(1, 1, -1).float())
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dynamics_weight = float(dynamics_weight)
        self.kinematic_weight = float(kinematic_weight)

    def compute_dynamics_residual(
        self,
        acceleration: torch.Tensor,
        generalized_force: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_pair(acceleration, generalized_force, "acceleration", "generalized_force")
        return acceleration - generalized_force / self.mass_vector

    def compute_kinematic_residuals(
        self,
        displacement: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._validate_triplet(displacement, velocity, acceleration)

        if displacement.shape[1] < 3:
            zero = displacement.new_zeros(1)
            return {
                "displacement": zero,
                "velocity": zero,
            }

        dt = self.dt
        displacement_residual = (
            displacement[:, 2:, :]
            - displacement[:, 1:-1, :]
            - velocity[:, 1:-1, :] * dt
            - (0.5 + self.alpha) * acceleration[:, 1:-1, :] * (dt**2)
            + self.alpha * acceleration[:, :-2, :] * (dt**2)
        )
        velocity_residual = (
            velocity[:, 2:, :]
            - velocity[:, 1:-1, :]
            - (1.0 + self.beta) * acceleration[:, 1:-1, :] * dt
            + self.beta * acceleration[:, :-2, :] * dt
        )
        return {
            "displacement": displacement_residual,
            "velocity": velocity_residual,
        }

    def forward(
        self,
        displacement: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        generalized_force: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dynamics_residual = self.compute_dynamics_residual(
            acceleration=acceleration,
            generalized_force=generalized_force,
        )
        kinematic_residuals = self.compute_kinematic_residuals(
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
        )

        loss_dynamics = dynamics_residual.abs().mean()
        loss_kinematics = (
            kinematic_residuals["displacement"].abs().mean()
            + kinematic_residuals["velocity"].abs().mean()
        )
        loss_total = (
            self.dynamics_weight * loss_dynamics
            + self.kinematic_weight * loss_kinematics
        )

        return {
            "dynamics_residual": dynamics_residual,
            "displacement_residual": kinematic_residuals["displacement"],
            "velocity_residual": kinematic_residuals["velocity"],
            "loss_dynamics": loss_dynamics,
            "loss_kinematics": loss_kinematics,
            "loss_total": loss_total,
        }

    def _validate_pair(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_name: str,
        right_name: str,
    ) -> None:
        if left.shape != right.shape:
            raise ValueError(
                f"{left_name} 与 {right_name} 形状必须一致，当前分别为 {left.shape} 和 {right.shape}。"
            )
        if left.ndim != 3:
            raise ValueError(f"{left_name} 必须是 [batch, time, dof] 三维张量。")
        if left.shape[-1] != self.mass_vector.shape[-1]:
            raise ValueError(
                f"末维自由度数应为 {self.mass_vector.shape[-1]}，当前为 {left.shape[-1]}。"
            )

    def _validate_triplet(
        self,
        displacement: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
    ) -> None:
        self._validate_pair(displacement, velocity, "displacement", "velocity")
        self._validate_pair(displacement, acceleration, "displacement", "acceleration")
