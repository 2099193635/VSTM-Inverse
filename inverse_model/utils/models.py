"""inverse_model/utils/models.py
可复用模型定义，供多个训练脚本共用。
"""
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).parent.parent.parent
for _p in (str(_ROOT / "physicsnemo"), str(Path(__file__).parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected


class VTCMDeepONet(torch.nn.Module):
    """
    1-D DeepONet：Branch(FNO) + Trunk(MLP) + Concat 融合。

    Branch : FNO  y [B, n_sensors, T]  ->  [B, latent_dim, T]
    Trunk  : MLP  t_norm [B, T, 1]    ->  [B, T, latent_dim]  (内部生成 linspace)
    Fuse   : cat([branch, trunk], -1)  ->  [B, T, 2*latent_dim]
    Proj   : Linear                    ->  [B, T, out_ch]
    Output : permute                   ->  [B, out_ch, T]
    """

    def __init__(self, branch_net: torch.nn.Module, trunk_net: torch.nn.Module,
                 out_channels: int, latent_dim: int):
        super().__init__()
        self.branch_net  = branch_net
        self.trunk_net   = trunk_net
        self.output_proj = torch.nn.Linear(2 * latent_dim, out_channels)
        torch.nn.init.normal_(self.output_proj.weight, std=0.01)
        torch.nn.init.zeros_(self.output_proj.bias)
        self.latent_dim = latent_dim

    def forward(self, y: torch.Tensor, x_query=None) -> torch.Tensor:
        B, _, T = y.shape
        branch_out = self.branch_net(y).permute(0, 2, 1)          # [B, T, latent]
        t_norm     = torch.linspace(0, 1, T, device=y.device, dtype=y.dtype)
        trunk_out  = self.trunk_net(t_norm.view(1, T, 1).expand(B, T, 1))  # [B, T, latent]
        out = self.output_proj(torch.cat([branch_out, trunk_out], dim=-1))  # [B, T, out_ch]
        return out.permute(0, 2, 1)                                # [B, out_ch, T]


def build_deeponet(in_channels: int, out_channels: int,
                   latent_channels: int = 128,
                   num_fno_layers: int = 8,
                   num_fno_modes: int = 64,
                   decoder_layers: int = 1,
                   decoder_layer_size: int = 128,
                   padding: int = 9,
                   trunk_layers: int = 4,
                   trunk_layer_size: int = 128) -> VTCMDeepONet:
    """工厂函数：构建 VTCMDeepONet 实例。"""
    branch = FNO(
        in_channels=in_channels, out_channels=latent_channels,
        decoder_layers=decoder_layers, decoder_layer_size=decoder_layer_size,
        dimension=1, latent_channels=latent_channels,
        num_fno_layers=num_fno_layers, num_fno_modes=num_fno_modes,
        padding=padding,
    )
    trunk = FullyConnected(
        in_features=1, out_features=latent_channels,
        layer_size=trunk_layer_size, num_layers=trunk_layers,
    )
    return VTCMDeepONet(branch, trunk, out_channels=out_channels, latent_dim=latent_channels)
