'''
Author: 2099193635 2099193635@qq.com
Date: 2026-04-22 06:50:24
LastEditors: 2099193635 2099193635@qq.com
LastEditTime: 2026-04-23 06:03:53
FilePath: /VTCM_PYTHON/inverse_model/train.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import argparse
import logging
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_DIR  = Path(__file__).parent.resolve()
_ROOT = _DIR.parent
sys.path.insert(0, str(_DIR))

from inverse_config import InverseConfig
from inverse_architecture import InverseOperator
from inverse_trainer import InverseTrainer


class InverseH5Dataset(Dataset):
    """从 inverse_dataset_gen 导出的 HDF5 读取样本。"""

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            self.y = np.asarray(f["y"], dtype=np.float32)              # [N, T, n_s]
            self.u = np.asarray(f["u"], dtype=np.float32)              # [N, L, n_dir]
            self.c = np.asarray(f["c"], dtype=np.float32)              # [N, n_cond]
            self.x_query = np.asarray(f["x_query"], dtype=np.float32)  # [N, L]
            self.seq_lengths = (
                np.asarray(f["seq_lengths"], dtype=np.int64)
                if "seq_lengths" in f
                else None
            )

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "y": torch.from_numpy(self.y[idx]),
            "u": torch.from_numpy(self.u[idx]),
            "c": torch.from_numpy(self.c[idx]),
            "x_query": torch.from_numpy(self.x_query[idx]),
        }
        if self.seq_lengths is not None:
            sample["seq_lengths"] = torch.tensor(self.seq_lengths[idx], dtype=torch.long)
        return sample

# logger config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_DIR / "train.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCNIO Training")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--trunk_layers", type=int,   default=8,
                   help="TrunkDecoder 的 MLP 层数，>=1")
    p.add_argument("--physics_mode", type=str,   default="pinn",
                   choices=["none", "frf", "pinn", "both"])
    p.add_argument("--ckpt_dir",     type=str,   default=str(_DIR / "checkpoints"))
    p.add_argument("--resume",       action="store_true",
                   help="Resume training from ckpt_dir/best_model.pt")
    p.add_argument("--dataset_dir",  type=str,   default=str(_ROOT / "datasets" / "VTCM_inverse"),
                   help="inverse_dataset_gen 导出的数据集目录（包含 train.hdf5 / validation.hdf5 / test.hdf5）")
    p.add_argument("--use_full_seq", action="store_true",
                   help="训练和验证都读取 *_full_seq.hdf5（兼容旧用法）")
    p.add_argument("--train-full-seq", dest="train_full_seq", action=argparse.BooleanOptionalAction,
                   default=False, help="训练是否使用 *_full_seq.hdf5，默认关闭（训练采用窗口数据）")
    p.add_argument("--val-full-seq", dest="val_full_seq", action=argparse.BooleanOptionalAction,
                   default=True, help="验证是否使用 *_full_seq.hdf5，默认开启（验证采用完整序列）")
    p.add_argument("--log_every",    type=int,   default=5)
    p.add_argument("--val_every",    type=int,   default=5,
                   help="每val_every个epoch进行一次验证和可视化")
    return p.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 主训练流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.use_full_seq:
        args.train_full_seq = True
        args.val_full_seq = True

    dataset_dir = Path(args.dataset_dir)
    ckpt_dir    = Path(args.ckpt_dir)

    # ── HDF5 数据集路径 ───────────────────────────────────────────────────
    train_suffix = "_full_seq" if args.train_full_seq else ""
    val_suffix = "_full_seq" if args.val_full_seq else ""
    train_h5 = dataset_dir / f"train{train_suffix}.hdf5"
    val_h5 = dataset_dir / f"validation{val_suffix}.hdf5"

    if not train_h5.exists():
        logger.error(f"Train dataset not found: {train_h5}")
        sys.exit(1)

    logger.info(f"Loading train dataset: {train_h5}")
    train_ds = InverseH5Dataset(train_h5)

    if val_h5.exists():
        logger.info(f"Loading val dataset: {val_h5}")
        val_ds = InverseH5Dataset(val_h5)
    else:
        logger.warning(f"Validation dataset not found: {val_h5}, running without val_loader.")
        val_ds = None

    if len(train_ds) == 0:
        logger.error("Train dataset is empty.")
        sys.exit(1)

    # 根据数据自动推断关键维度
    sample0 = train_ds[0]
    n_sensors = int(sample0["y"].shape[-1])
    spatial_len = int(sample0["x_query"].shape[0])
    n_cond = int(sample0["c"].shape[-1])

    # ── 配置 ────────────────────────────────────────────────────────────────
    cfg = InverseConfig(
        n_sensors    = n_sensors,
        n_cond       = n_cond,
        spatial_len  = spatial_len,
        trunk_layers = args.trunk_layers,
        physics_mode = args.physics_mode,
    )

    logger.info(
        f"\nConfig: epochs={args.epochs}, batch={args.batch_size}, "
        f"physics={args.physics_mode}, trunk_layers={args.trunk_layers}, "
        f"train_full_seq={args.train_full_seq}, val_full_seq={args.val_full_seq}\n"
        f"  spatial_len={cfg.spatial_len}, n_sensors={cfg.n_sensors}, n_cond={cfg.n_cond}"
    )

    if val_ds is not None:
        val_spatial_len = int(val_ds[0]["x_query"].shape[0])
        logger.info(f"Sequence length | train={spatial_len}, val={val_spatial_len}")

    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds is not None else 0}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    ) if (val_ds is not None and len(val_ds) > 0) else None

    # ── 模型 ─────────────────────────────────────────────────────────────────
    model = InverseOperator(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    if args.resume and (ckpt_dir / "best_model.pt").exists():
        logger.info(f"Resuming from {ckpt_dir / 'best_model.pt'}")
        trainer = InverseTrainer.load_checkpoint(
            ckpt_dir / "best_model.pt", model, cfg, device=device
        )
    else:
        trainer = InverseTrainer(model, cfg, device=device)
    
    # ── 训练 ─────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Training PCNIO  |  {args.epochs} epochs  |  {device}")
    logger.info(f"{'='*60}")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        ckpt_dir=ckpt_dir,
        log_every=args.log_every,
        val_every=args.val_every,
    )

    # ── 保存 loss 曲线 ───────────────────────────────────────────────────────
    hist_path = ckpt_dir / "train_history.npz"
    np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})
    logger.info(f"Loss history saved: {hist_path}")

    # ── 打印最终指标 ─────────────────────────────────────────────────────────
    final_train = history.get("train_l_total", [float("nan")])[-1]
    final_val   = history.get("val_l_total",   [float("nan")])[-1]
    logger.info(f"\nFinal  train_loss={final_train:.6f}  val_loss={final_val:.6f}")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")

    # ── 快速绘制 loss 曲线（可选）────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        for key in ["train_l_total", "val_l_total",
                    "train_l_data",  "train_l_frf", "train_l_spec"]:
            if key in history and any(np.isfinite(history[key])):
                ax.plot(history[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("PCNIO Training Loss")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = ckpt_dir / "loss_curve.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()
        logger.info(f"Loss curve saved: {fig_path}")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
