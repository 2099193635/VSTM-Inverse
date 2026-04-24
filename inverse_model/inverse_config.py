from dataclasses import dataclass, field
from typing import List

@dataclass
class InverseConfig:
    # input dimension, default 3 (vehicle Z, front bogie Z, rear bogie Z)
    n_sensors: int = 3

    # Condition variable dimensions: 8 orders of natural frequencies + 8 orders of damping ratios + 1 operating speed.
    n_cond: int = 17

    # number of irregularity directions, default 1 (vertical)
    n_directions: int = 1

    # highperameter for architectures
    # width of hidden layers
    width: int = 64

    branch_modes: int = 32
    
    branch_layers: int = 4

    trunk_hidden: int = 128

    trunk_layers: int = 3

    fno_modes: int = 16
    
    fpe_freqs: int = 8

    activation: str = "gelu"

    # training hyperparameters

    lambda_phys: float = 0.1

    lambda_spec: float = 0.0

    lambda_spatial: float = 1.0

    lambda_corr: float = 2.0

    lambda_grad: float = 0.2

    lambda_std: float = 1.0

    dropout: float = 0.0

    lr: float = 1e-3

    lr_gamma: float = 0.9999

    # data
    spatial_len: int = 256

    track_grade: int = 2

    n_modal: int = 8

    physics_mode: str = "frf"
    # physics_mode: 
    # "none": no physics loss
    # "frf": use FRF loss, only for vertical direction
    # "pinn": PINN track pde residual loss
    # "both": combine both FRF and PINN losses

    lambda_frf: float = 0.1

    lambda_pinn: float = 0.01

    frf_learnable: bool = True
    # whether to permit the frf modal residue to be learnable

    # track structure parameters
    track_EI: float = 1.5e6    # track flexural rigidity, in N*m^2
