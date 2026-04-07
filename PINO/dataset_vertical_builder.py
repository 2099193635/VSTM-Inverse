"""
不改动 PINO/dataset.py 的前提下，构建“仅垂向自由度”的训练/测试 DataLoader。

核心思路：
1) 先复用原始 create_train_test_dataloaders() 得到标准数据集（含归一化）；
2) 再通过 Dataset 包装器筛选 output 通道：
   - 默认保留车辆 35DOF 中 Z 对应索引 [1,6,11,16,21,26,31]
   - 默认拼接 disp/vel/acc => 7*3=21 通道
3) 可选返回 init_state（每个窗口首时刻状态）供模型显式喂初值。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PINO.dataset import VTCMDataset, create_train_test_dataloaders, _build_physics_context


# 车辆 35DOF 中 4 个轮对垂向 Z 索引（0-based）
WHEELSET_Z_DOF_INDICES: Tuple[int, ...] = (16, 21, 26, 31)


@dataclass
class VerticalDatasetConfig:
    train_dir: str
    test_dir: str
    batch_size: int = 4
    seq_length: int = 256
    stride: int = 128
    shuffle: bool = True
    device: str = "cpu"

    # 车辆 35DOF 中垂向 Z 索引（0-based）
    z_dof_indices: Tuple[int, ...] = (1, 6, 11, 16, 21, 26, 31)

    # 选择输出分量：disp/vel/acc 任意组合
    components: Tuple[str, ...] = ("disp", "vel", "acc")

    # 是否在每个样本返回首时刻初值 init_state: [C_selected]
    return_init_state: bool = True

    # 导出目录（参考 datasets/Darcy_241 的组织方式）
    export_dir: str = "/workspace/VTCM_PYTHON/datasets/VTCM_vertical"

    # 是否额外生成 validation.hdf5（与 test.hdf5 同内容，便于兼容 Darcy_241 习惯）
    write_validation_alias: bool = True

    # 单一来源切分模式：若设置该目录，则忽略 train_dir/test_dir，按 split_ratio 切分 train/val/test
    source_dir: str = ""
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 42

    # 是否同时生成完整序列数据集（每文件一个样本，跳过初始瞬态后 pad 到最长）
    # 输出为 train_full_seq.hdf5 / validation_full_seq.hdf5 / test_full_seq.hdf5
    include_full_sequences: bool = True
    # 初始跳过时长（秒），与切片数据集保持一致
    skip_initial_seconds: float = 2.0


class ChannelSelectDataset(Dataset):
    """对原始样本进行输出通道筛选，不改动底层数据集。"""

    def __init__(self, base_dataset: Dataset, channel_indices: Sequence[int], return_init_state: bool = True):
        self.base_dataset = base_dataset
        self.channel_indices = torch.as_tensor(channel_indices, dtype=torch.long)
        self.return_init_state = return_init_state

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.base_dataset[idx]
        y = sample["output"]  # [C_full, T]
        y_sel = y.index_select(dim=0, index=self.channel_indices.to(y.device))

        out = dict(sample)
        out["output"] = y_sel
        if self.return_init_state:
            out["init_state"] = y_sel[:, 0].clone()  # [C_selected]
        return out


def _build_channel_indices(
    dof: int = 35,
    z_dof_indices: Sequence[int] = (1, 6, 11, 16, 21, 26, 31),
    components: Sequence[str] = ("disp", "vel", "acc"),
    force_include_wheelset_z: bool = True,
) -> List[int]:
    comp_to_offset = {"disp": 0, "vel": dof, "acc": 2 * dof}

    # 统一并保证轮对垂向 DOF 在 output 中（不单独另存 dataset）
    z_idx = [int(i) for i in z_dof_indices]
    if force_include_wheelset_z:
        for wi in WHEELSET_Z_DOF_INDICES:
            if wi not in z_idx:
                z_idx.append(int(wi))

    indices: List[int] = []
    for c in components:
        c = c.lower().strip()
        if c not in comp_to_offset:
            raise ValueError(f"不支持的 components 项: {c}，可选 disp/vel/acc")
        offset = comp_to_offset[c]
        indices.extend([offset + int(i) for i in z_idx])
    return indices


def _slice_norm_params(norm_params: Dict, channel_indices: Sequence[int]) -> Dict:
    """从 full-DOF 归一化参数切片出 selected 通道参数。"""
    out = dict(norm_params)
    x_mean = norm_params.get("X_mean")
    x_std = norm_params.get("X_std")
    if x_mean is not None and x_std is not None:
        ch = np.asarray(channel_indices, dtype=np.int64)
        out["X_mean"] = np.asarray(x_mean)[:, ch]
        out["X_std"] = np.asarray(x_std)[:, ch]
    out["selected_channel_indices"] = list(channel_indices)
    return out


def _extract_numeric_params(ctx_dict: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """从参数子字典中提取所有数值字段，返回有序的 (field_names, values) 对。"""
    names: List[str] = []
    vals: List[float] = []
    for k, v in ctx_dict.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            names.append(k)
            vals.append(float(v))
        elif isinstance(v, (np.floating, np.integer)):
            names.append(k)
            vals.append(float(v))
    return names, vals


def _collect_physics_arrays(
    contexts: List[Dict[str, Any]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    从 physics_context 列表中提取各子组参数，构建 [N, n_params] 数组。

    子组映射：
      vehicle  → vehicle_params  （车体/转向架/轮对质量、刚度、阻尼等）
      rail     → rail_params     （钢轨截面、弹性模量等）
      fastener → fastener_params （扣件刚度/阻尼）
      subrail  → subrail_params  （轨下结构参数）
      line     → line_params     （线路几何参数）

    同时提取 run_metadata.vx_set → vx_mps（km/h 转 m/s）。

    Returns:
        arrays:      {"vehicle_params": ndarray[N,p], ..., "vx_mps": ndarray[N]}
        field_names: {"vehicle_params": ["Mc_base", ...], ...}
    """
    GROUPS = ["vehicle", "rail", "fastener", "subrail", "line"]
    N = len(contexts)

    # 确定各组规范字段顺序（取第一个非空 context）
    canonical: Dict[str, List[str]] = {}
    for g in GROUPS:
        for ctx in contexts:
            sub = ctx.get(g, {}) if isinstance(ctx, dict) else {}
            if isinstance(sub, dict) and sub:
                names, _ = _extract_numeric_params(sub)
                if names:
                    canonical[g] = names
                    break

    arrays: Dict[str, np.ndarray] = {}
    field_names: Dict[str, List[str]] = {}

    for g in GROUPS:
        fnames = canonical.get(g, [])
        if not fnames:
            continue
        key = f"{g}_params"
        arr = np.full((N, len(fnames)), np.nan, dtype=np.float32)
        for i, ctx in enumerate(contexts):
            sub = ctx.get(g, {}) if isinstance(ctx, dict) else {}
            if not isinstance(sub, dict):
                continue
            for j, fname in enumerate(fnames):
                v = sub.get(fname, np.nan)
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float, np.floating, np.integer)):
                    arr[i, j] = float(v)
        arrays[key] = arr
        field_names[key] = fnames

    # 车速：run_metadata.vx_set (km/h) → vx_mps (m/s)
    vx = np.full(N, np.nan, dtype=np.float32)
    for i, ctx in enumerate(contexts):
        meta = ctx.get("run_metadata", {}) if isinstance(ctx, dict) else {}
        vx_kmh = meta.get("vx_set") if isinstance(meta, dict) else None
        if vx_kmh is not None and isinstance(vx_kmh, (int, float)):
            vx[i] = float(vx_kmh) / 3.6
    arrays["vx_mps"] = vx
    field_names["vx_mps"] = ["vx_mps"]

    return arrays, field_names


def _compute_norm_from_dataset(dataset: VTCMDataset) -> Dict:
    """当上游未执行归一化时，在本模块内回退计算归一化参数。"""
    if len(dataset.data) == 0:
        return {"X_mean": None, "X_std": None, "u_mean": None, "u_std": None}

    u_all = np.concatenate([d["u"] for d in dataset.data], axis=0)
    y_all = np.concatenate([d["y"] for d in dataset.data], axis=0)
    if u_all.ndim == 2:
        u_mean = u_all.mean(axis=0, keepdims=True)
        u_std = u_all.std(axis=0, keepdims=True) + 1e-8
    else:
        u_mean = float(u_all.mean())
        u_std = float(u_all.std() + 1e-8)
    return {
        "u_mean": u_mean,
        "u_std": u_std,
        "X_mean": y_all.mean(axis=0, keepdims=True),
        "X_std": y_all.std(axis=0, keepdims=True) + 1e-8,
    }


def _apply_norm_to_dataset(dataset: VTCMDataset, norm_params: Dict) -> None:
    """将给定归一化参数应用到 dataset.data。"""
    if len(dataset.data) == 0:
        return
    u_mean = norm_params["u_mean"]
    u_std = norm_params["u_std"]
    x_mean = norm_params["X_mean"]
    x_std = norm_params["X_std"]
    for d in dataset.data:
        d["u"] = (d["u"] - u_mean) / u_std
        d["y"] = (d["y"] - x_mean) / x_std


def create_vertical_train_test_dataloaders(
    cfg: VerticalDatasetConfig,
):
    """
    构建“仅垂向自由度”训练/测试 DataLoader。

    Returns:
      train_loader, test_loader, norm_params_selected
    """
    try:
        train_loader_full, test_loader_full, norm_params_full = create_train_test_dataloaders(
            train_dir=cfg.train_dir,
            test_dir=cfg.test_dir,
            batch_size=cfg.batch_size,
            seq_length=cfg.seq_length,
            stride=cfg.stride,
            shuffle=cfg.shuffle,
            device=cfg.device,
        )
    except Exception:
        # 回退路径：不依赖 PINO/dataset.py 内部归一化状态，显式重建并归一化
        train_npz_files = list(Path(cfg.train_dir).rglob("simulation_result.npz"))
        test_npz_files = list(Path(cfg.test_dir).rglob("simulation_result.npz"))
        if len(train_npz_files) == 0:
            raise ValueError(f"未找到训练集 NPZ 文件: {cfg.train_dir}")
        if len(test_npz_files) == 0:
            raise ValueError(f"未找到测试集 NPZ 文件: {cfg.test_dir}")

        train_dataset_full = VTCMDataset(
            npz_files=train_npz_files,
            seq_length=cfg.seq_length,
            stride=cfg.stride,
            normalize=False,
            device=cfg.device,
        )
        test_dataset_full = VTCMDataset(
            npz_files=test_npz_files,
            seq_length=cfg.seq_length,
            stride=cfg.stride,
            normalize=False,
            device=cfg.device,
        )
        norm_params_full = _compute_norm_from_dataset(train_dataset_full)
        _apply_norm_to_dataset(train_dataset_full, norm_params_full)
        _apply_norm_to_dataset(test_dataset_full, norm_params_full)

        train_loader_full = DataLoader(
            train_dataset_full,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=0,
        )
        test_loader_full = DataLoader(
            test_dataset_full,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

    channel_indices = _build_channel_indices(
        dof=35,
        z_dof_indices=cfg.z_dof_indices,
        components=cfg.components,
    )

    train_ds = ChannelSelectDataset(
        base_dataset=train_loader_full.dataset,
        channel_indices=channel_indices,
        return_init_state=cfg.return_init_state,
    )
    test_ds = ChannelSelectDataset(
        base_dataset=test_loader_full.dataset,
        channel_indices=channel_indices,
        return_init_state=cfg.return_init_state,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    norm_params_sel = _slice_norm_params(norm_params_full, channel_indices)
    return train_loader, test_loader, norm_params_sel


def _split_npz_files(
    npz_files: List[Path],
    ratio: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """按文件粒度切分 NPZ 列表为 train / val / test，与切片数据集使用相同随机种子。"""
    rng = np.random.default_rng(seed)
    shuffled = list(npz_files)
    rng.shuffle(shuffled)

    n = len(shuffled)
    r_train, r_val, r_test = [r / sum(ratio) for r in ratio]
    n_train = int(n * r_train)
    n_val   = int(n * r_val)
    n_test  = n - n_train - n_val

    train = shuffled[:n_train]
    val   = shuffled[n_train: n_train + n_val]
    test  = shuffled[n_train + n_val:]
    print(f"  [文件级切分] 共 {n} 个文件 → train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def _materialize_full_sequences_from_npz(
    npz_files: List[Path],
    channel_indices: List[int],
    norm_params: Dict,
    skip_initial_seconds: float = 2.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    从 NPZ 文件列表构建完整序列数据集（每文件一个样本）。

    - 跳过初始瞬态（skip_initial_seconds）
    - 应用与切片数据集相同的归一化参数
    - 不同文件若长度不同则 zero-pad 到最大长度，同时记录各文件实际长度 seq_lengths
    - 输出通道与切片数据集一致（channel_indices 相同）

    Returns:
        arrays:     {input: (N,2,T_max), output: (N,C,T_max),
                     seq_lengths: (N,), dt: (N,), source_file: (N,),
                     vehicle_params, rail_params, ...}
        phys_fields: {vehicle_params: [...], ...}
    """
    ch = np.asarray(channel_indices, dtype=np.int64)

    # 归一化参数整形为 1D 便于广播
    u_mean = np.asarray(norm_params["u_mean"], dtype=np.float32).reshape(-1)   # (2,)
    u_std  = np.asarray(norm_params["u_std"],  dtype=np.float32).reshape(-1)   # (2,)
    X_mean = np.asarray(norm_params["X_mean"], dtype=np.float32).reshape(-1)   # (C,)
    X_std  = np.asarray(norm_params["X_std"],  dtype=np.float32).reshape(-1)   # (C,)

    inputs_list:   List[np.ndarray] = []
    outputs_list:  List[np.ndarray] = []
    seq_lengths:   List[int]        = []
    dts:           List[float]      = []
    source_files:  List[str]        = []
    phys_contexts: List[Dict]       = []

    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
            X  = data.get("X")
            V  = data.get("V")
            A  = data.get("A")
            dt = float(data.get("dt", 1e-4))

            if X is None or V is None or A is None:
                print(f"  [跳过] {npz_path}: 缺少 X/V/A")
                continue

            Nt = X.shape[0]
            skip_steps = int(skip_initial_seconds / dt)

            if Nt <= skip_steps:
                print(f"  [跳过] {npz_path}: 序列太短 {Nt} <= skip_steps {skip_steps}")
                continue

            # ── 构建输入激励 [Nt, 2] ──
            irre_L = data.get("Irre_bz_L_ref")
            irre_R = data.get("Irre_bz_R_ref")
            if irre_L is not None and irre_R is not None:
                u_seq = np.stack([
                    np.asarray(irre_L[:Nt], dtype=np.float32),
                    np.asarray(irre_R[:Nt], dtype=np.float32),
                ], axis=1)
            elif irre_L is not None:
                uL = np.asarray(irre_L[:Nt], dtype=np.float32)
                u_seq = np.stack([uL, np.zeros_like(uL)], axis=1)
            elif irre_R is not None:
                uR = np.asarray(irre_R[:Nt], dtype=np.float32)
                u_seq = np.stack([np.zeros_like(uR), uR], axis=1)
            else:
                u_seq = np.zeros((Nt, 2), dtype=np.float32)

            # ── 构建输出响应，选通道 [Nt, C_sel] ──
            y_seq = np.concatenate([
                X.reshape(Nt, -1),
                V.reshape(Nt, -1),
                A.reshape(Nt, -1),
            ], axis=1).astype(np.float32)[:, ch]

            # ── 跳过初始瞬态 ──
            u_seq = u_seq[skip_steps:]   # [T, 2]
            y_seq = y_seq[skip_steps:]   # [T, C_sel]
            T = u_seq.shape[0]

            # ── 归一化（与切片数据集共享参数）──
            u_seq = (u_seq - u_mean) / u_std   # [T, 2]
            y_seq = (y_seq - X_mean) / X_std   # [T, C_sel]

            # ── 转置为 [C, T]（与切片格式一致）──
            inputs_list.append(u_seq.T.astype(np.float32))   # [2, T]
            outputs_list.append(y_seq.T.astype(np.float32))  # [C, T]
            seq_lengths.append(T)
            dts.append(dt)
            source_files.append(str(npz_path))
            phys_contexts.append(_build_physics_context(npz_path))

        except Exception as exc:
            print(f"  [错误] {npz_path}: {exc}")

    if len(inputs_list) == 0:
        raise ValueError("_materialize_full_sequences_from_npz: 没有成功加载任何 NPZ 文件")

    # ── Zero-pad 到最大长度 ──
    T_max  = max(seq_lengths)
    N      = len(inputs_list)
    C_in   = inputs_list[0].shape[0]
    C_out  = outputs_list[0].shape[0]

    inp_arr = np.zeros((N, C_in,  T_max), dtype=np.float32)
    out_arr = np.zeros((N, C_out, T_max), dtype=np.float32)
    for i, (u, y, T) in enumerate(zip(inputs_list, outputs_list, seq_lengths)):
        inp_arr[i, :, :T] = u
        out_arr[i, :, :T] = y

    arrays: Dict[str, np.ndarray] = {
        "input":       inp_arr,
        "output":      out_arr,
        "seq_lengths": np.asarray(seq_lengths, dtype=np.int32),
        "dt":          np.asarray(dts, dtype=np.float32),
        "source_file": np.asarray(source_files, dtype=object),
    }

    phys_arrays, phys_fields = _collect_physics_arrays(phys_contexts)
    arrays.update(phys_arrays)

    print(f"  完整序列物化完成：{N} 个文件，T_max={T_max}，"
          f"input {inp_arr.shape}，output {out_arr.shape}")
    return arrays, phys_fields


def _build_split_lengths(total: int, ratio: Tuple[float, float, float]) -> Tuple[int, int, int]:
    r_train, r_val, r_test = ratio
    if min(r_train, r_val, r_test) < 0:
        raise ValueError(f"split_ratio 不能为负数: {ratio}")
    s = r_train + r_val + r_test
    if s <= 0:
        raise ValueError(f"split_ratio 之和必须 > 0: {ratio}")

    r_train, r_val, r_test = r_train / s, r_val / s, r_test / s
    n_train = int(total * r_train)
    n_val = int(total * r_val)
    n_test = total - n_train - n_val

    # 防止极小样本下某一项为负
    if n_test < 0:
        n_test = 0
        n_val = max(0, total - n_train)
    return n_train, n_val, n_test


def create_vertical_train_val_test_from_single_source(
    cfg: VerticalDatasetConfig,
):
    """从单一目录加载并按比例切分 train/val/test。"""
    source_dir = Path(cfg.source_dir)
    npz_files = list(source_dir.rglob("simulation_result.npz"))
    if len(npz_files) == 0:
        raise ValueError(f"未找到任何 NPZ 文件，请检查目录: {source_dir}")

    base_dataset = VTCMDataset(
        npz_files=npz_files,
        seq_length=cfg.seq_length,
        stride=cfg.stride,
        normalize=False,
        device=cfg.device,
    )
    norm_params_full = _compute_norm_from_dataset(base_dataset)
    _apply_norm_to_dataset(base_dataset, norm_params_full)

    channel_indices = _build_channel_indices(
        dof=35,
        z_dof_indices=cfg.z_dof_indices,
        components=cfg.components,
    )
    selected_dataset = ChannelSelectDataset(
        base_dataset=base_dataset,
        channel_indices=channel_indices,
        return_init_state=cfg.return_init_state,
    )

    n_total = len(selected_dataset)
    n_train, n_val, n_test = _build_split_lengths(n_total, cfg.split_ratio)
    gen = torch.Generator().manual_seed(cfg.split_seed)
    train_ds, val_ds, test_ds = random_split(selected_dataset, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    norm_params_sel = _slice_norm_params(norm_params_full, channel_indices)
    return train_loader, val_loader, test_loader, norm_params_sel


def denormalize_output_selected(
    y_norm: Union[np.ndarray, torch.Tensor],
    norm_params_selected: Dict,
) -> Union[np.ndarray, torch.Tensor]:
    """按 selected 通道参数反归一化，支持 [B,C,T] 或 [C,T]。"""
    mean = norm_params_selected.get("X_mean")
    std = norm_params_selected.get("X_std")
    if mean is None or std is None:
        return y_norm

    if isinstance(y_norm, torch.Tensor):
        mean_t = torch.as_tensor(mean, dtype=y_norm.dtype, device=y_norm.device).squeeze(0)
        std_t = torch.as_tensor(std, dtype=y_norm.dtype, device=y_norm.device).squeeze(0)
        if y_norm.dim() == 3:
            return y_norm * std_t.view(1, -1, 1) + mean_t.view(1, -1, 1)
        if y_norm.dim() == 2:
            return y_norm * std_t.view(-1, 1) + mean_t.view(-1, 1)
        raise ValueError(f"不支持的维度: {tuple(y_norm.shape)}")

    arr = np.asarray(y_norm)
    mean_arr = np.asarray(mean, dtype=arr.dtype).reshape(-1)
    std_arr = np.asarray(std, dtype=arr.dtype).reshape(-1)
    if arr.ndim == 3:
        return arr * std_arr.reshape(1, -1, 1) + mean_arr.reshape(1, -1, 1)
    if arr.ndim == 2:
        return arr * std_arr.reshape(-1, 1) + mean_arr.reshape(-1, 1)
    raise ValueError(f"不支持的维度: {arr.shape}")


def _materialize_dataset(
    dataset: Dataset,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    将 map-style dataset 物化为 numpy 数组，便于一次性写入 HDF5。

    同时从每个样本的 physics_context 中提取车体参数、下部结构参数等。

    Returns:
        arrays:       {input, output, init_state, dt, source_file,
                       window_start, window_end,
                       vehicle_params, rail_params, fastener_params,
                       subrail_params, line_params, vx_mps}
        phys_fields:  {vehicle_params: ["Mc_base", ...], ...}
    """
    inputs: List[np.ndarray] = []
    outputs: List[np.ndarray] = []
    init_states: List[np.ndarray] = []
    dts: List[np.float32] = []
    source_files: List[str] = []
    win_starts: List[np.int32] = []
    win_ends: List[np.int32] = []
    physics_contexts: List[Dict[str, Any]] = []

    has_init_state = False
    for i in range(len(dataset)):
        sample = dataset[i]
        inputs.append(sample["input"].detach().cpu().numpy().astype(np.float32))
        outputs.append(sample["output"].detach().cpu().numpy().astype(np.float32))
        dts.append(np.float32(sample.get("dt", 0.0)))
        source_files.append(str(sample.get("source_file", "")))
        win_starts.append(np.int32(sample.get("window_start", 0)))
        win_ends.append(np.int32(sample.get("window_end", 0)))
        ctx = sample.get("physics_context", {})
        physics_contexts.append(ctx if isinstance(ctx, dict) else {})
        if "init_state" in sample:
            has_init_state = True
            init_states.append(sample["init_state"].detach().cpu().numpy().astype(np.float32))

    out: Dict[str, np.ndarray] = {
        "input": np.stack(inputs, axis=0),      # [N, 1, T]
        "output": np.stack(outputs, axis=0),    # [N, C, T]
        "dt": np.asarray(dts, dtype=np.float32),
        "window_start": np.asarray(win_starts, dtype=np.int32),
        "window_end": np.asarray(win_ends, dtype=np.int32),
        # 保留为 Python 字符串列表，写 HDF5 时按 utf-8 可变长字符串处理
        "source_file": np.asarray(source_files, dtype=object),
    }
    if has_init_state and len(init_states) == len(inputs):
        out["init_state"] = np.stack(init_states, axis=0)  # [N, C]

    # 提取物理参数（vehicle/rail/fastener/subrail/line + vx_mps）
    phys_arrays, phys_fields = _collect_physics_arrays(physics_contexts)
    out.update(phys_arrays)

    return out, phys_fields


def _write_hdf5(
    file_path: Path,
    arrays: Dict[str, np.ndarray],
    attrs: Dict[str, Union[str, int, float]],
    dataset_attrs: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """
    将 numpy 数组写入 HDF5。

    - attrs:         写到文件根级别的属性（格式、seq_length 等）
    - dataset_attrs: 写到各 dataset 上的属性，如物理参数字段名
      格式: {dataset_key: {attr_name: attr_value}}
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        for k, v in arrays.items():
            if k == "source_file":
                str_dtype = h5py.string_dtype(encoding="utf-8")
                ds = f.create_dataset(k, data=np.asarray(v, dtype=object), dtype=str_dtype)
            else:
                ds = f.create_dataset(k, data=v, compression="gzip")
            # 写入 per-dataset 属性（如 fields="Mc_base,Mt_base,..."）
            if dataset_attrs and k in dataset_attrs:
                for ak, av in dataset_attrs[k].items():
                    ds.attrs[ak] = av
        for ak, av in attrs.items():
            f.attrs[ak] = av


def export_vertical_datasets_to_hdf5(cfg: VerticalDatasetConfig):
    """
    构建垂向数据集并导出到 datasets/<name>/train.hdf5、validation.hdf5、test.hdf5。
    额外保存 norm_stats.npz 与 meta.json。

    HDF5 中除响应数据外，还包含以下物理参数 dataset：
      - vehicle_params  [N, p_veh]   车辆参数（车体/转向架/轮对质量、刚度、阻尼等）
      - rail_params     [N, p_rail]  钢轨参数
      - fastener_params [N, p_fas]   扣件参数
      - subrail_params  [N, p_sub]   轨下结构参数
      - line_params     [N, p_line]  线路几何参数
      - vx_mps          [N]          运行速度 (m/s)
    每个 *_params dataset 均附有 attrs["fields"] 记录列名（逗号分隔）。
    """
    use_single_source = bool(str(cfg.source_dir).strip())
    if use_single_source:
        train_loader, val_loader, test_loader, norm_sel = create_vertical_train_val_test_from_single_source(cfg)
        train_arrays, train_phys_fields = _materialize_dataset(train_loader.dataset)
        val_arrays,  _                  = _materialize_dataset(val_loader.dataset)
        test_arrays, _                  = _materialize_dataset(test_loader.dataset)
    else:
        train_loader, test_loader, norm_sel = create_vertical_train_test_dataloaders(cfg)
        train_arrays, train_phys_fields = _materialize_dataset(train_loader.dataset)
        test_arrays, _                  = _materialize_dataset(test_loader.dataset)
        val_arrays = test_arrays

    export_dir = Path(cfg.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # 根级属性
    attrs = {
        "format": "VTCM_vertical_time_series_v1",
        "seq_length": int(cfg.seq_length),
        "stride": int(cfg.stride),
        "components": ",".join(cfg.components),
        "z_dof_indices": ",".join(str(i) for i in cfg.z_dof_indices),
    }

    # per-dataset 属性：物理参数字段名列表
    dataset_attrs: Dict[str, Dict[str, str]] = {}
    for key, fnames in train_phys_fields.items():
        if key != "vx_mps" and fnames:
            dataset_attrs[key] = {"fields": ",".join(fnames)}

    # ── 写入切片数据集（windowed）──
    train_h5 = export_dir / "train.hdf5"
    test_h5  = export_dir / "test.hdf5"
    _write_hdf5(train_h5, train_arrays, attrs, dataset_attrs)
    _write_hdf5(test_h5,  test_arrays,  attrs, dataset_attrs)

    if use_single_source:
        _write_hdf5(export_dir / "validation.hdf5", val_arrays, attrs, dataset_attrs)
    elif cfg.write_validation_alias:
        _write_hdf5(export_dir / "validation.hdf5", test_arrays, attrs, dataset_attrs)

    # 保存归一化参数（selected channels）—— 切片与完整序列共享
    np.savez(
        export_dir / "norm_stats.npz",
        X_mean=np.asarray(norm_sel.get("X_mean")) if norm_sel.get("X_mean") is not None else np.array([]),
        X_std=np.asarray(norm_sel.get("X_std"))   if norm_sel.get("X_std")  is not None else np.array([]),
        u_mean=np.asarray(norm_sel.get("u_mean")) if norm_sel.get("u_mean") is not None else np.array([]),
        u_std=np.asarray(norm_sel.get("u_std"))   if norm_sel.get("u_std")  is not None else np.array([]),
        selected_channel_indices=np.asarray(norm_sel.get("selected_channel_indices", []), dtype=np.int64),
    )

    phys_field_meta = {k: v for k, v in train_phys_fields.items() if k != "vx_mps"}

    # ── 写入完整序列数据集（full_seq）──
    full_seq_info: Dict[str, Any] = {}
    if cfg.include_full_sequences:
        print("\n[完整序列] 开始物化完整序列数据集 ...")
        channel_indices = _build_channel_indices(
            dof=35,
            z_dof_indices=cfg.z_dof_indices,
            components=cfg.components,
        )
        # 确保 norm_sel 含 u_mean/u_std/X_mean/X_std
        full_norm = {
            "u_mean": norm_sel.get("u_mean"),
            "u_std":  norm_sel.get("u_std"),
            "X_mean": norm_sel.get("X_mean"),
            "X_std":  norm_sel.get("X_std"),
        }

        # 根据模式确定各分割的 NPZ 文件列表
        if use_single_source:
            all_npz = list(Path(cfg.source_dir).rglob("simulation_result.npz"))
            fs_train_npz, fs_val_npz, fs_test_npz = _split_npz_files(
                all_npz, cfg.split_ratio, cfg.split_seed
            )
        else:
            fs_train_npz = list(Path(cfg.train_dir).rglob("simulation_result.npz"))
            fs_test_npz  = list(Path(cfg.test_dir).rglob("simulation_result.npz"))
            fs_val_npz   = fs_test_npz  # train_dir/test_dir 模式 val = test

        # 全序列 HDF5 根级属性（标注 format 为 full_seq，记录 T_max 等）
        full_attrs_base = {
            "format": "VTCM_vertical_full_seq_v1",
            "skip_initial_seconds": cfg.skip_initial_seconds,
            "components": ",".join(cfg.components),
            "z_dof_indices": ",".join(str(i) for i in cfg.z_dof_indices),
        }

        def _export_full_split(npz_list: List[Path], tag: str) -> Optional[Dict[str, np.ndarray]]:
            if not npz_list:
                print(f"  [跳过] {tag}: NPZ 列表为空")
                return None
            print(f"  [{tag}] 物化 {len(npz_list)} 个文件 ...")
            arr, pf = _materialize_full_sequences_from_npz(
                npz_list, channel_indices, full_norm, cfg.skip_initial_seconds
            )
            full_attrs = dict(full_attrs_base)
            full_attrs["T_max"] = int(arr["seq_lengths"].max())
            da: Dict[str, Dict[str, str]] = {}
            for key, fnames in pf.items():
                if key != "vx_mps" and fnames:
                    da[key] = {"fields": ",".join(fnames)}
            _write_hdf5(export_dir / f"{tag}_full_seq.hdf5", arr, full_attrs, da)
            print(f"  [{tag}] 已写入 {export_dir / f'{tag}_full_seq.hdf5'}")
            return arr

        fs_train_arr = _export_full_split(fs_train_npz, "train")
        fs_val_arr   = _export_full_split(fs_val_npz,   "validation")
        fs_test_arr  = _export_full_split(fs_test_npz,  "test")

        full_seq_info = {
            "train_full_seq_hdf5":  str(export_dir / "train_full_seq.hdf5"),
            "val_full_seq_hdf5":    str(export_dir / "validation_full_seq.hdf5"),
            "test_full_seq_hdf5":   str(export_dir / "test_full_seq.hdf5"),
            "train_full_seq_files": len(fs_train_npz),
            "val_full_seq_files":   len(fs_val_npz),
            "test_full_seq_files":  len(fs_test_npz),
            "T_max": int(fs_train_arr["seq_lengths"].max()) if fs_train_arr is not None else 0,
        }

    # ── 保存元信息 ──
    meta: Dict[str, Any] = {
        "train_dir": cfg.train_dir,
        "test_dir": cfg.test_dir,
        "source_dir": cfg.source_dir,
        "split_ratio": list(cfg.split_ratio),
        "seq_length": cfg.seq_length,
        "stride": cfg.stride,
        "skip_initial_seconds": cfg.skip_initial_seconds,
        "components": list(cfg.components),
        "z_dof_indices": list(cfg.z_dof_indices),
        "return_init_state": cfg.return_init_state,
        "export_dir": str(export_dir),
        # 切片数据集统计
        "windowed": {
            "train_samples": int(train_arrays["input"].shape[0]),
            "val_samples":   int(val_arrays["input"].shape[0]),
            "test_samples":  int(test_arrays["input"].shape[0]),
            "input_shape":   list(train_arrays["input"].shape[1:]),
            "output_shape":  list(train_arrays["output"].shape[1:]),
            "files": ["train.hdf5", "validation.hdf5", "test.hdf5"],
        },
        "physics_fields": phys_field_meta,
    }
    if full_seq_info:
        meta["full_seq"] = full_seq_info

    (export_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = {
        "export_dir": str(export_dir),
        # 切片数据集
        "train_hdf5":    str(train_h5),
        "val_hdf5":      str(export_dir / "validation.hdf5"),
        "test_hdf5":     str(test_h5),
        "train_samples": int(train_arrays["input"].shape[0]),
        "val_samples":   int(val_arrays["input"].shape[0]),
        "test_samples":  int(test_arrays["input"].shape[0]),
        "output_channels": int(train_arrays["output"].shape[1]),
        "physics_fields":  phys_field_meta,
    }
    result.update(full_seq_info)
    return result


if __name__ == "__main__":
    cfg = VerticalDatasetConfig(
        train_dir="/workspace/VTCM_PYTHON/results/pino_train_dataset",
        test_dir="/workspace/VTCM_PYTHON/results/pino_test_dataset",
        batch_size=2,
        seq_length=256,
        stride=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
        components=("disp", "vel", "acc"),  # 输出 21 通道
        return_init_state=True,
        export_dir="/workspace/VTCM_PYTHON/datasets/VTCM_vertical",
        write_validation_alias=True,
        source_dir="/workspace/VTCM_PYTHON/results/pino_train_dataset",
        split_ratio=(0.8, 0.1, 0.1),
        split_seed=42,
        include_full_sequences=True,  # 同时生成完整序列数据集
        skip_initial_seconds=2.0,
    )

    info = export_vertical_datasets_to_hdf5(cfg)
    print("\n=== 导出完成 ===")
    print("── 切片数据集（windowed）──")
    print(f"  train.hdf5:      {info['train_samples']} 个窗口")
    print(f"  validation.hdf5: {info['val_samples']} 个窗口")
    print(f"  test.hdf5:       {info['test_samples']} 个窗口")
    if "train_full_seq_hdf5" in info:
        print("── 完整序列数据集（full_seq）──")
        print(f"  train_full_seq.hdf5:      {info['train_full_seq_files']} 个仿真文件")
        print(f"  validation_full_seq.hdf5: {info['val_full_seq_files']} 个仿真文件")
        print(f"  test_full_seq.hdf5:       {info['test_full_seq_files']} 个仿真文件")
        print(f"  T_max = {info['T_max']} 步")
    print(f"\n输出目录: {info['export_dir']}")
