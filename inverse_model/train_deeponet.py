"""
train_deeponet.py
DeepONet (Branch=FNO + Trunk=MLP) 纯数据驱动基线训练脚本。
与 train_fno.py 结构一致，仅模型替换为 VTCMDeepONet。
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_DIR  = Path(__file__).parent
_ROOT = _DIR.parent

for _p in (str(_ROOT / "physicsnemo"), str(_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (
    InverseH5Dataset, build_target,
    weighted_data_loss, z0_spectral_loss,
    plot_val_predictions, plot_inference_samples,
    EarlyStopping, save_best_checkpoint, resume_checkpoint,
    build_deeponet,
)


# ── 参数解析 ──────────────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepONet Baseline (data-only) Training")
    p.add_argument("--epochs",             type=int,   default=200)
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--lr",                 type=float, default=3e-4)
    p.add_argument("--gamma",              type=float, default=0.999487)
    p.add_argument("--grad_clip_norm",     type=float, default=1.0)
    # 数据损失
    p.add_argument("--z0_data_weight",     type=float, default=5.0)
    p.add_argument("--w24_data_weight",    type=float, default=2.0)
    p.add_argument("--use_spectral_loss",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--spectral_weight",    type=float, default=0.5)
    p.add_argument("--spectral_hf_boost",  type=float, default=4.0)
    # 模型结构
    p.add_argument("--latent_channels",    type=int,   default=128)
    p.add_argument("--num_fno_layers",     type=int,   default=8)
    p.add_argument("--num_fno_modes",      type=int,   default=64)
    p.add_argument("--decoder_layers",     type=int,   default=1)
    p.add_argument("--decoder_layer_size", type=int,   default=128)
    p.add_argument("--padding",            type=int,   default=9)
    p.add_argument("--trunk_layers",       type=int,   default=4)
    p.add_argument("--trunk_layer_size",   type=int,   default=128)
    # 路径
    p.add_argument("--ckpt_dir",    type=str, default=str(_DIR / "checkpoints_deeponet_baseline"))
    p.add_argument("--fig_dir",     type=str, default=str(_DIR / "figures_deeponet_baseline"))
    p.add_argument("--dataset_dir", type=str, default=str(_ROOT / "datasets" / "VTCM_inverse"))
    p.add_argument("--vis_every",   type=int, default=10)
    # 数据集
    p.add_argument("--train-full-seq", dest="train_full_seq", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-full-seq",   dest="val_full_seq",   action=argparse.BooleanOptionalAction, default=True)
    # 训练控制
    p.add_argument("--resume",              action="store_true", default=False)
    p.add_argument("--early_stop_patience", type=int, default=15)
    return p.parse_args(argv)


# ── 验证 ──────────────────────────────────────────────────────────────────────
def validate(model, loader, device, out_norm_stats, args, epoch: int) -> float:
    model.eval()
    total, n = 0.0, 0
    viz_pred = viz_tgt = None
    with torch.no_grad():
        for batch in loader:
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)
            target = build_target(batch, device)
            pred   = model(inp)

            ld = weighted_data_loss(pred, target, args.z0_data_weight, args.w24_data_weight)
            if args.use_spectral_loss:
                ld = ld + args.spectral_weight * z0_spectral_loss(
                    pred, target, hf_boost=args.spectral_hf_boost)
            total += ld.item(); n += 1
            if viz_pred is None:
                viz_pred = pred[0].detach().cpu().numpy()
                viz_tgt  = target[0].detach().cpu().numpy()

    if viz_pred is not None and epoch % args.vis_every == 0:
        plot_val_predictions(viz_pred, viz_tgt, out_norm_stats, args.fig_dir, epoch)

    return total / max(n, 1)


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = Path(args.dataset_dir)
    tsuf = "_full_seq" if args.train_full_seq else ""
    vsuf = "_full_seq" if args.val_full_seq   else ""
    train_ds = InverseH5Dataset(dataset_dir / f"train{tsuf}.hdf5")
    val_ds   = InverseH5Dataset(dataset_dir / f"validation{vsuf}.hdf5",
                                out_norm_stats=train_ds.out_norm_stats)

    _kw = dict(num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **_kw) \
                   if len(val_ds) > 0 else None

    out_norm_stats = train_ds.out_norm_stats
    in_ch  = train_ds[0]["y"].shape[-1]
    out_ch = train_ds[0]["out_norm"].shape[-1]

    print(f"[DeepONet Baseline] Device={device}  train={len(train_ds)}  val={len(val_ds)}"
          f"  in_ch={in_ch}  out_ch={out_ch}")

    model = build_deeponet(
        in_channels=in_ch, out_channels=out_ch,
        latent_channels=args.latent_channels,
        num_fno_layers=args.num_fno_layers, num_fno_modes=args.num_fno_modes,
        decoder_layers=args.decoder_layers, decoder_layer_size=args.decoder_layer_size,
        padding=args.padding,
        trunk_layers=args.trunk_layers, trunk_layer_size=args.trunk_layer_size,
    ).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    early_stop = EarlyStopping(patience=args.early_stop_patience) \
                 if args.early_stop_patience > 0 else None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.fig_dir,  exist_ok=True)

    start_epoch = global_step = 0
    if args.resume:
        start_epoch, global_step, _ = resume_checkpoint(
            args.ckpt_dir, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, n_upd = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{args.epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)
            target = build_target(batch, device)
            pred   = model(inp)

            ld = weighted_data_loss(pred, target, args.z0_data_weight, args.w24_data_weight)
            if args.use_spectral_loss:
                ld = ld + args.spectral_weight * z0_spectral_loss(
                    pred, target, hf_boost=args.spectral_hf_boost)

            if not torch.isfinite(ld):
                global_step += 1; continue

            ld.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1; epoch_loss += ld.item(); n_upd += 1
            pbar.set_postfix(loss=f"{ld.item():.3e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_train = epoch_loss / max(n_upd, 1)
        val_loss = 0.0
        if val_loader is not None:
            val_loss = validate(model, val_loader, device, out_norm_stats, args, epoch)

        print(f"[{epoch + 1:4d}/{args.epochs}] train={avg_train:.4e}"
              f"  val={val_loss:.4e}"
              f"  lr={optimizer.param_groups[0]['lr']:.3e}")

        is_best = save_best_checkpoint(
            args.ckpt_dir, model, optimizer, scheduler,
            epoch, val_loss, global_step)
        if is_best:
            print(f"  ✓ New best val={val_loss:.4e}")

        if early_stop is not None and val_loader is not None:
            if early_stop.step(val_loss):
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    if val_loader is not None:
        print("Generating inference plots ...")
        model.eval()
        pred_list, tgt_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(pred_list) >= 8: break
                inp    = batch["y"].to(device, non_blocking=True).transpose(1, 2)
                target = build_target(batch, device)
                pred   = model(inp)
                for i in range(min(pred.shape[0], 8 - len(pred_list))):
                    pred_list.append(pred[i].cpu().numpy())
                    tgt_list.append(target[i].cpu().numpy())
        plot_inference_samples(pred_list, tgt_list, out_norm_stats,
                               args.fig_dir, args.epochs - 1)
        print(f"Figures → {args.fig_dir}")


if __name__ == "__main__":
    main()
