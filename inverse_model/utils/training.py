"""inverse_model/utils/training.py
训练辅助：早停、最优 checkpoint、线性爬坡调度、checkpoint 恢复。
"""
import json
import os
import shutil
from pathlib import Path

import torch


# ── 线性爬坡调度 ──────────────────────────────────────────────────────────────
def linear_ramp(step: int, start_step: int, ramp_steps: int,
                max_value: float = 1.0) -> float:
    """从 start_step 开始，在 ramp_steps 步内线性爬坡到 max_value。"""
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return float(max_value)
    p = min((step - start_step) / float(ramp_steps), 1.0)
    return float(max_value) * p


def physics_weight_at_step(step: int, warmup_steps: int, ramp_steps: int,
                            max_weight: float) -> float:
    return linear_ramp(step, warmup_steps, ramp_steps, max_weight)


def deriv_mix_alpha_at_step(step: int, warmup_steps: int, blend_steps: int,
                             max_ratio: float) -> float:
    return linear_ramp(step, warmup_steps, blend_steps, max_ratio)


# ── 最优 checkpoint（只保留单一最佳文件）────────────────────────────────────
_BEST_CKPT_NAME  = "best_model.pt"
_BEST_META_NAME  = "best_meta.json"


def save_best_checkpoint(ckpt_dir: str | Path, model: torch.nn.Module,
                         optimizer, scheduler, epoch: int,
                         val_loss: float, global_step: int) -> bool:
    """
    若 val_loss 优于历史最优，则覆盖保存 best_model.pt，返回 True；否则返回 False。
    同时将 epoch / val_loss / global_step 写入 best_meta.json。
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta_path = ckpt_dir / _BEST_META_NAME
    best_val  = float("inf")
    if meta_path.exists():
        with open(meta_path) as f:
            best_val = json.load(f).get("val_loss", float("inf"))

    if val_loss >= best_val:
        return False

    torch.save({
        "epoch":       epoch,
        "global_step": global_step,
        "val_loss":    val_loss,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
    }, ckpt_dir / _BEST_CKPT_NAME)

    with open(meta_path, "w") as f:
        json.dump({"epoch": epoch, "global_step": global_step, "val_loss": val_loss}, f, indent=2)

    return True


def resume_checkpoint(ckpt_dir: str | Path, model: torch.nn.Module,
                      optimizer, scheduler,
                      device: torch.device) -> tuple[int, int, float]:
    """
    从 best_model.pt 恢复模型、优化器、调度器状态。
    返回 (epoch, global_step, best_val_loss)。
    若无 checkpoint 则返回 (0, 0, inf)。
    """
    ckpt_path = Path(ckpt_dir) / _BEST_CKPT_NAME
    if not ckpt_path.exists():
        return 0, 0, float("inf")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    epoch       = int(ckpt.get("epoch",       0))
    global_step = int(ckpt.get("global_step", 0))
    best_val    = float(ckpt.get("val_loss",  float("inf")))
    print(f"[Resume] Loaded checkpoint: epoch={epoch + 1}, global_step={global_step}, val_loss={best_val:.4e}")
    return epoch + 1, global_step, best_val


# ── 早停 ──────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    若验证损失在 patience 个 epoch 内没有改善（降低 > min_delta），则触发停止。

    Args:
        patience:  容忍不改善的 epoch 数。
        min_delta: 视为改善的最小下降量。
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")

    def step(self, val_loss: float) -> bool:
        """传入当前验证损失，返回是否应停止训练。"""
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def state_dict(self) -> dict:
        return {"counter": self.counter, "best": self.best}

    def load_state_dict(self, d: dict):
        self.counter = int(d.get("counter", 0))
        self.best    = float(d.get("best", float("inf")))
