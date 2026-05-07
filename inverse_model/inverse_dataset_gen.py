"""
inverse_dataset_gen.py
======================
为逆向模型（InverseOperator）生成 HDF5 训练数据集。

输入：仿真结果目录（递归搜索 simulation_result.npz）
输出：datasets/VTCM_inverse/{train,validation,test}.hdf5 + norm_stats.npz + meta.json

HDF5 每个 split 含以下 dataset：
  y              [N, T, n_sensors]     传感器加速度（200Hz，降采样，z 方向）
  u              [N, L, n_irr_dofs]    轨道不平顺（空间域）
    phys_x         [N, T, 7]             物理约束所需垂向位移 [Zc,Zt1,Zt2,Zw1..Zw4]
    phys_v         [N, T, 7]             物理约束所需垂向速度 [Zc,Zt1,Zt2,Zw1..Zw4]
    phys_a         [N, T, 7]             物理约束所需垂向加速度 [Zc,Zt1,Zt2,Zw1..Zw4]
    phys_z0        [N, T, 1]             参考轨道垂向不平顺（与 y 时间对齐）
  c              [N, n_cond]           条件向量（固有频率+阻尼比+车速）
  x_query        [N, L]                空间坐标（[0,1] 归一化）
  vx             [N]                   运行速度 m/s
  natural_freqs  [N, n_modal]
  damping_ratios [N, n_modal]
  source_file    [N]                   原始 npz 路径（utf-8 字符串）
  window_start   [N]                   时间窗起始步（target_freq 下）
  window_end     [N]                   时间窗结束步

根级 attrs 记录格式版本与超参数。
norm_stats.npz 保存 y/u 的均值和标准差（按 train split 计算）。

用法
----
python inverse_dataset_gen.py \\
    --source_dir  /workspace/VTCM_PYTHON/results/pino_train_dataset \\
    --export_dir  /workspace/VTCM_PYTHON/datasets/VTCM_inverse \\
    --split_ratio 0.8 0.1 0.1 \\
    --window_size 256 \\
    --stride      128 \\
    --target_freq 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# ── 项目路径 ─────────────────────────────────────────────────────────────────
_DIR  = Path(__file__).resolve().parent
_ROOT = _DIR.parent
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_ROOT))

from inverse_config import InverseConfig
from inverse_dataset import compute_cond_vecotr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inverse_dataset_gen")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class InverseDatasetGenConfig:
    # ── 数据源 ──────────────────────────────────────────────────────────────
    # source_dir 非空时：按 split_ratio 自动切分 train/val/test
    source_dir:  str = str(_ROOT / "results" / "pino_train_dataset")
    # 也可分别指定 train/test（source_dir 为空时生效）
    train_dir:   str = str(_ROOT / "results" / "pino_train_dataset")
    test_dir:    str = str(_ROOT / "results" / "pino_test_dataset")

    export_dir:  str = str(_ROOT / "datasets" / "VTCM_inverse")

    # 切分比例 (train / val / test)
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed:  int = 42

    # 采样频率
    sim_freq:    int = 10_000   # Hz，仿真原始采样率
    target_freq: int = 10000      # Hz，降采样后目标采样率

    # 滑窗参数
    window_size: int = 256      # 时间点数（target_freq 下）
    stride:      int = 128      # 步长

    # 逆向模型超参
    n_modal:     int = 8
    spatial_len: int = 256      # 不平顺空间采样点数（对齐 window_size）

    # 传感器 DOFs（在车辆子系统本地索引中，0-based）
    # # 默认取车体Z=1, 前构架Z=6, 后构架Z=11
    # sensor_local_dofs: Tuple[int, ...] = (1, 6, 11)
    sensor_local_dofs: Tuple[int, ...] = (1,)
    # 车辆子系统列数（从 idx_car_start 开始取几列）
    n_car_cols: int = 20

    # 不平顺方向：0=垂向，1=横向
    irr_dofs: Tuple[int, ...] = (0,)

    # 初始瞬态跳过时长（秒）
    skip_initial_seconds: float = 2.0

    # 是否额外写 validation.hdf5 别名（当使用 train_dir/test_dir 模式时）
    write_validation_alias: bool = True

    # 是否同时导出 full_seq（每个 npz 一个样本，按最长序列 zero-pad）
    include_full_sequences: bool = True

    # physics_informed.py 需要的垂向 DOFs（车辆局部 0-based）
    # [Zc, Zt1, Zt2, Zw1, Zw2, Zw3, Zw4]
    physics_local_dofs: Tuple[int, ...] = (1, 6, 11, 16, 21, 26, 31)

    # 轮对间距参数（用于计算各轮 Z₀ 超前延迟，避免运行时 time-shift 边界填充）
    # Lc: 车辆半轴距（车体中心到转向架中心距离，m）
    # Lt: 转向架半轴距（转向架中心到轮对距离，m）
    # 空间顺序：Xw1 = Xw4 + 2*(Lc+Lt)，Xw2 = Xw4 + 2*Lc，Xw3 = Xw4 + 2*Lt
    Lc: float = 9.0    # 车辆半轴距 (m)
    Lt: float = 1.2    # 转向架半轴距 (m)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _read_vx(json_path: Path, data) -> float:
    """从 argparse_params.json 或 npz 中读取运行速度（m/s）。"""
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        return float(params.get("vx_set", 215.0)) / 3.6
    if "vx" in data:
        vx = data["vx"]
        v  = float(np.mean(vx)) if hasattr(vx, "__len__") else float(vx)
        return v / 3.6 if v > 100.0 else v
    return 215.0 / 3.6


def _split_files(
    files: List[Path],
    ratio: Tuple[float, float, float],
    seed:  int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)

    s = sum(ratio)
    r_tr, r_va, r_te = ratio[0] / s, ratio[1] / s, ratio[2] / s
    n = len(shuffled)
    n_tr = int(n * r_tr)
    n_va = int(n * r_va)
    n_te = n - n_tr - n_va
    if n_te < 0:
        n_te = 0
        n_va = max(0, n - n_tr)

    tr = shuffled[:n_tr]
    va = shuffled[n_tr: n_tr + n_va]
    te = shuffled[n_tr + n_va:]
    logger.info(f"文件切分 → train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te


def _write_hdf5(
    file_path:  Path,
    arrays:     Dict[str, np.ndarray],
    root_attrs: Dict,
) -> None:
    """将 numpy 数组字典写入 HDF5 文件（gzip 压缩）。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        for k, v in arrays.items():
            if k == "source_file":
                str_dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(k, data=np.asarray(v, dtype=object), dtype=str_dt)
            else:
                f.create_dataset(k, data=v, compression="gzip")
        for ak, av in root_attrs.items():
            f.attrs[ak] = av
    logger.info(f"  已写入 {file_path}  ({arrays['y'].shape[0]} 个窗口)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 核心：单个 npz → 滑窗记录列表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _windows_from_npz(
    npz_path: Path,
    cfg:      InverseDatasetGenConfig,
) -> List[Dict[str, np.ndarray]]:
    """从单个 npz 提取所有滑窗样本，返回 list of dict。"""
    json_path  = npz_path.parent / "argparse_params.json"
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        logger.warning(f"[skip] {npz_path}: {e}")
        return []

    ds_factor = cfg.sim_freq // cfg.target_freq

    # ── 加速度（车辆子系统） ──────────────────────────────────────────────
    A_raw = data.get("A")
    if A_raw is None:
        logger.warning(f"[skip] {npz_path}: 缺少 'A' 字段")
        return []

    X_raw = data.get("X")
    V_raw = data.get("V")
    if X_raw is None or V_raw is None:
        logger.warning(f"[skip] {npz_path}: 缺少 'X' 或 'V' 字段")
        return []

    T_sim, n_dof      = A_raw.shape
    idx_car_start     = int(data["idx_car_start"]) if "idx_car_start" in data else 0
    col_end           = min(idx_car_start + cfg.n_car_cols, n_dof)
    acc_full          = A_raw[::ds_factor, idx_car_start:col_end].astype(np.float32)  # [T_ds, n_car]
    T_ds              = acc_full.shape[0]

    # physics_informed 所需状态量（按固定局部 DOF 索引）
    p_dofs = list(cfg.physics_local_dofs)
    valid_p_dofs = [d for d in p_dofs if d < n_dof]
    if len(valid_p_dofs) != len(p_dofs):
        logger.warning(f"[skip] {npz_path}: physics_local_dofs 超界，n_dof={n_dof}, dofs={p_dofs}")
        return []

    phys_x_full = X_raw[::ds_factor, valid_p_dofs].astype(np.float32)  # [T_ds, 7]
    phys_v_full = V_raw[::ds_factor, valid_p_dofs].astype(np.float32)  # [T_ds, 7]
    phys_a_full = A_raw[::ds_factor, valid_p_dofs].astype(np.float32)  # [T_ds, 7]

    # ── 轨道不平顺（降采样） ──────────────────────────────────────────────
    irr_dict: Dict[int, np.ndarray] = {}
    for d in cfg.irr_dofs:
        if d == 0:
            irr_L = data.get("Irre_bz_L_ref")
            irr_R = data.get("Irre_bz_R_ref")
        else:
            irr_L = data.get("Irre_by_L_ref")
            irr_R = data.get("Irre_by_R_ref")

        if irr_L is not None and irr_R is not None:
            raw = 0.5 * (np.asarray(irr_L[:T_sim], dtype=np.float64) +
                         np.asarray(irr_R[:T_sim], dtype=np.float64))
        elif irr_L is not None:
            raw = np.asarray(irr_L[:T_sim], dtype=np.float64)
        elif irr_R is not None:
            raw = np.asarray(irr_R[:T_sim], dtype=np.float64)
        else:
            raw = np.zeros(T_sim, dtype=np.float64)
        irr_dict[d] = raw[::ds_factor].astype(np.float32)  # [T_ds]

    # 垂向基准不平顺（physics_informed 里的 Z0）
    irr_bz_L = data.get("Irre_bz_L_ref")
    irr_bz_R = data.get("Irre_bz_R_ref")
    if irr_bz_L is not None and irr_bz_R is not None:
        z0_raw = 0.5 * (np.asarray(irr_bz_L[:T_sim], dtype=np.float64) +
                        np.asarray(irr_bz_R[:T_sim], dtype=np.float64))
    elif irr_bz_L is not None:
        z0_raw = np.asarray(irr_bz_L[:T_sim], dtype=np.float64)
    elif irr_bz_R is not None:
        z0_raw = np.asarray(irr_bz_R[:T_sim], dtype=np.float64)
    else:
        z0_raw = np.zeros(T_sim, dtype=np.float64)
    z0_full = z0_raw[::ds_factor].astype(np.float32)  # [T_ds]

    # ── 车速 & 条件向量 ───────────────────────────────────────────────────
    vx_mps      = _read_vx(json_path, data)

    # ── 各轮对 Z₀ 超前延迟步数（整数，在 target_freq 域）───────────────────
    # 轮位顺序：Xw1（最前）= Xw4 + 2*(Lc+Lt)，以此类推
    # 因此 Z₀_wi(t) = z0_full[t + d_wi_int]，在数据生成时直接切片，消除边界填充
    _dt_ds = 1.0 / cfg.target_freq
    _d_w1_int = int(round(2.0 * (cfg.Lc + cfg.Lt) / vx_mps / _dt_ds))
    _d_w2_int = int(round(2.0 * cfg.Lc              / vx_mps / _dt_ds))
    _d_w3_int = int(round(2.0 * cfg.Lt              / vx_mps / _dt_ds))
    _z0_delays = {"w1": _d_w1_int, "w2": _d_w2_int, "w3": _d_w3_int, "w4": 0}
    _max_delay  = _d_w1_int  # w1 延迟最大

    n_freqs_raw = data["natural_freqs"].astype(np.float32)  if "natural_freqs"  in data else np.zeros(cfg.n_modal, dtype=np.float32)
    n_damps_raw = data["damping_ratios"].astype(np.float32) if "damping_ratios" in data else np.zeros(cfg.n_modal, dtype=np.float32)
    freqs = np.zeros(cfg.n_modal, dtype=np.float32)
    damps = np.zeros(cfg.n_modal, dtype=np.float32)
    n = min(cfg.n_modal, len(n_freqs_raw))
    freqs[:n] = n_freqs_raw[:n]
    damps[:n] = n_damps_raw[:n]

    cond_vec = compute_cond_vecotr(freqs, damps, float(vx_mps), cfg.n_modal)  # [n_cond]

    # ── 初始瞬态跳过 ──────────────────────────────────────────────────────
    skip_steps = int(cfg.skip_initial_seconds * cfg.target_freq)
    irr_len    = min(v.shape[0] for v in irr_dict.values())

    # ── 滑窗 ──────────────────────────────────────────────────────────────
    W = cfg.window_size
    S = cfg.stride
    L = cfg.spatial_len
    sensor_dofs = list(cfg.sensor_local_dofs)
    n_sensors   = len(sensor_dofs)
    x_query     = np.linspace(0.0, 1.0, L, dtype=np.float32)

    records: List[Dict[str, np.ndarray]] = []
    for start in range(skip_steps, T_ds - W + 1, S):
        end = start + W

        # 跳过末尾无法完整覆盖 w1 延迟的窗口（避免边界填充误差）
        if end + _max_delay > len(z0_full):
            break

        # 传感器加速度 [W, n_sensors]
        y_win = np.zeros((W, n_sensors), dtype=np.float32)
        for i, col in enumerate(sensor_dofs):
            if col < acc_full.shape[1]:
                y_win[:, i] = acc_full[start:end, col]

            # physics_informed 所需状态 [W, 7]
            phys_x_win = phys_x_full[start:end, :]
            phys_v_win = phys_v_full[start:end, :]
            phys_a_win = phys_a_full[start:end, :]
            phys_z0_win = z0_full[start:end].reshape(-1, 1)  # [W, 1] wheel-4 基准（向后兼容保留）

        # 各轮对独立 Z₀ 切片（真实轨道激励，无边界填充）[W, 1]
        phys_z0_w4_win = z0_full[start                  : end                 ].reshape(-1, 1)
        phys_z0_w3_win = z0_full[start + _d_w3_int      : end + _d_w3_int     ].reshape(-1, 1)
        phys_z0_w2_win = z0_full[start + _d_w2_int      : end + _d_w2_int     ].reshape(-1, 1)
        phys_z0_w1_win = z0_full[start + _d_w1_int      : end + _d_w1_int     ].reshape(-1, 1)

        # 空间域不平顺 [L, n_irr_dofs]
        # 这里采用与时间窗“同索引对齐”：
        # acc_full 与 irr_dict 均由同一原始序列按 ds_factor 降采样得到，
        # 因此直接使用 start/end 对齐，避免再次用 vx*dt 映射导致采样偏移。
        x_start = start
        x_end   = x_start + L
        if x_end > irr_len:
            break

        u_win = np.stack(
            [irr_dict[d][x_start:x_end] for d in cfg.irr_dofs], axis=-1
        )  # [L, n_irr_dofs]

        records.append({
            "y":              y_win,              # [T, n_sensors]
            "u":              u_win,              # [L, n_irr_dofs]
            "phys_x":         phys_x_win,         # [T, 7]
            "phys_v":         phys_v_win,         # [T, 7]
            "phys_a":         phys_a_win,         # [T, 7]
            "phys_z0":        phys_z0_win,        # [T, 1]  wheel-4 基准（向后兼容）
            "phys_z0_w1":     phys_z0_w1_win,     # [T, 1]  轮1真实 Z₀（超前 d_w1 步）
            "phys_z0_w2":     phys_z0_w2_win,     # [T, 1]  轮2真实 Z₀
            "phys_z0_w3":     phys_z0_w3_win,     # [T, 1]  轮3真实 Z₀
            "phys_z0_w4":     phys_z0_w4_win,     # [T, 1]  轮4真实 Z₀（同 phys_z0）
            "c":              cond_vec.copy(),    # [n_cond]
            "x_query":        x_query,            # [L]
            "vx":             np.atleast_1d(np.float32(vx_mps)),       # [1]
            "natural_freqs":  freqs.copy(),       # [n_modal]
            "damping_ratios": damps.copy(),        # [n_modal]
            "window_start":   np.atleast_1d(np.int32(start)),
            "window_end":     np.atleast_1d(np.int32(end)),
            "source_file":    str(npz_path),
        })

    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 全量物化：文件列表 → 合并数组字典
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _materialize(
    files: List[Path],
    cfg:   InverseDatasetGenConfig,
    split: str,
) -> Optional[Dict[str, np.ndarray]]:
    """从文件列表提取所有窗口，合并为 numpy 数组字典。"""
    all_records: List[Dict] = []
    for npz_path in files:
        recs = _windows_from_npz(npz_path, cfg)
        all_records.extend(recs)

    if not all_records:
        logger.warning(f"[{split}] 没有有效窗口，跳过。")
        return None

    logger.info(f"[{split}] {len(all_records)} 个窗口，来自 {len(files)} 个文件。")

    # 逐 key 合并
    arrays: Dict[str, np.ndarray] = {}
    sample_keys = [k for k in all_records[0] if k != "source_file"]
    for k in sample_keys:
        arrays[k] = np.stack([r[k] for r in all_records], axis=0)
    arrays["source_file"] = np.asarray([r["source_file"] for r in all_records], dtype=object)

    return arrays


def _materialize_full_sequences(
    files: List[Path],
    cfg: InverseDatasetGenConfig,
    split: str,
    norm_stats: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    full_seq 模式：每个 npz 对应一个样本。
    - 跳过每个文件前 skip_initial_seconds
    - 按最长序列 zero-pad
    - 可选应用与 windowed 相同的归一化参数（y/u）
    """
    ds_factor = cfg.sim_freq // cfg.target_freq
    skip_steps = int(cfg.skip_initial_seconds * cfg.target_freq)

    y_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    phys_x_list: List[np.ndarray] = []
    phys_v_list: List[np.ndarray] = []
    phys_a_list: List[np.ndarray] = []
    phys_z0_list: List[np.ndarray] = []
    phys_z0_w1_list: List[np.ndarray] = []
    phys_z0_w2_list: List[np.ndarray] = []
    phys_z0_w3_list: List[np.ndarray] = []
    phys_z0_w4_list: List[np.ndarray] = []
    xq_list: List[np.ndarray] = []
    c_list: List[np.ndarray] = []
    vx_list: List[np.ndarray] = []
    nf_list: List[np.ndarray] = []
    dr_list: List[np.ndarray] = []
    src_list: List[str] = []
    seq_len_list: List[int] = []

    sensor_dofs = list(cfg.sensor_local_dofs)
    n_sensors = len(sensor_dofs)

    for npz_path in files:
        json_path = npz_path.parent / "argparse_params.json"
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            logger.warning(f"[full_seq][skip] {npz_path}: {e}")
            continue

        A_raw = data.get("A")
        X_raw = data.get("X")
        V_raw = data.get("V")
        if A_raw is None or X_raw is None or V_raw is None:
            logger.warning(f"[full_seq][skip] {npz_path}: 缺少 'A'/'X'/'V' 字段")
            continue

        T_sim, n_dof = A_raw.shape
        idx_car_start = int(data["idx_car_start"]) if "idx_car_start" in data else 0
        col_end = min(idx_car_start + cfg.n_car_cols, n_dof)
        acc_full = A_raw[::ds_factor, idx_car_start:col_end].astype(np.float32)  # [T_ds, n_car]

        p_dofs = list(cfg.physics_local_dofs)
        valid_p_dofs = [d for d in p_dofs if d < n_dof]
        if len(valid_p_dofs) != len(p_dofs):
            logger.warning(f"[full_seq][skip] {npz_path}: physics_local_dofs 超界，n_dof={n_dof}, dofs={p_dofs}")
            continue
        phys_x_full = X_raw[::ds_factor, valid_p_dofs].astype(np.float32)  # [T_ds, 7]
        phys_v_full = V_raw[::ds_factor, valid_p_dofs].astype(np.float32)  # [T_ds, 7]
        phys_a_full = A_raw[::ds_factor, valid_p_dofs].astype(np.float32)  # [T_ds, 7]

        # 构建不平顺序列（降采样）
        irr_dict: Dict[int, np.ndarray] = {}
        for d in cfg.irr_dofs:
            if d == 0:
                irr_L = data.get("Irre_bz_L_ref")
                irr_R = data.get("Irre_bz_R_ref")
            else:
                irr_L = data.get("Irre_by_L_ref")
                irr_R = data.get("Irre_by_R_ref")

            if irr_L is not None and irr_R is not None:
                raw = 0.5 * (np.asarray(irr_L[:T_sim], dtype=np.float64) +
                             np.asarray(irr_R[:T_sim], dtype=np.float64))
            elif irr_L is not None:
                raw = np.asarray(irr_L[:T_sim], dtype=np.float64)
            elif irr_R is not None:
                raw = np.asarray(irr_R[:T_sim], dtype=np.float64)
            else:
                raw = np.zeros(T_sim, dtype=np.float64)
            irr_dict[d] = raw[::ds_factor].astype(np.float32)

        irr_bz_L = data.get("Irre_bz_L_ref")
        irr_bz_R = data.get("Irre_bz_R_ref")
        if irr_bz_L is not None and irr_bz_R is not None:
            z0_raw = 0.5 * (np.asarray(irr_bz_L[:T_sim], dtype=np.float64) +
                            np.asarray(irr_bz_R[:T_sim], dtype=np.float64))
        elif irr_bz_L is not None:
            z0_raw = np.asarray(irr_bz_L[:T_sim], dtype=np.float64)
        elif irr_bz_R is not None:
            z0_raw = np.asarray(irr_bz_R[:T_sim], dtype=np.float64)
        else:
            z0_raw = np.zeros(T_sim, dtype=np.float64)
        z0_full = z0_raw[::ds_factor].astype(np.float32)  # [T_ds]

        irr_len = min(v.shape[0] for v in irr_dict.values())
        T_ds = acc_full.shape[0]
        T_use = min(T_ds, irr_len) - skip_steps
        if T_use <= 1:
            logger.warning(f"[full_seq][skip] {npz_path}: 有效长度不足，T_use={T_use}")
            continue

        start = skip_steps
        end = start + T_use

        # ── 车速 ──────────────────────────────────────────────────────
        vx_mps = _read_vx(json_path, data)

        # ── 各轮对 Z₀ 超前延迟（target_freq 域整数步）──────────────────────
        _dt_ds_fs   = 1.0 / cfg.target_freq
        _d_w1_fs    = int(round(2.0 * (cfg.Lc + cfg.Lt) / vx_mps / _dt_ds_fs))
        _d_w2_fs    = int(round(2.0 * cfg.Lc              / vx_mps / _dt_ds_fs))
        _d_w3_fs    = int(round(2.0 * cfg.Lt              / vx_mps / _dt_ds_fs))
        _z0_len_fs  = len(z0_full)

        def _z0_slice_fs(delay: int) -> np.ndarray:
            """cut z0_full[start+delay:end+delay]，末尾超界部分用 edge-pad（全序列说明用）。"""
            s, e = start + delay, end + delay
            if e <= _z0_len_fs:
                return z0_full[s:e].reshape(-1, 1)
            # 超界尾部用边界匹配（edge padding）
            valid = min(max(_z0_len_fs - s, 0), T_use)
            arr = np.empty((T_use, 1), dtype=np.float32)
            if valid > 0:
                arr[:valid, 0] = z0_full[s:s + valid]
                arr[valid:, 0] = z0_full[_z0_len_fs - 1]
            else:
                arr[:, 0] = z0_full[_z0_len_fs - 1]
            return arr

        # y 序列 [T, n_sensors]
        y_seq = np.zeros((T_use, n_sensors), dtype=np.float32)
        for i, col in enumerate(sensor_dofs):
            if col < acc_full.shape[1]:
                y_seq[:, i] = acc_full[start:end, col]

            phys_x_seq = phys_x_full[start:end, :]                      # [T, 7]
            phys_v_seq = phys_v_full[start:end, :]                      # [T, 7]
            phys_a_seq = phys_a_full[start:end, :]                      # [T, 7]
            phys_z0_seq = z0_full[start:end].reshape(-1, 1).astype(np.float32)  # [T, 1]

        # 各轮对独立 Z₀ 序列（不平顺如实轨道激励）
        phys_z0_w4_seq = _z0_slice_fs(0)
        phys_z0_w3_seq = _z0_slice_fs(_d_w3_fs)
        phys_z0_w2_seq = _z0_slice_fs(_d_w2_fs)
        phys_z0_w1_seq = _z0_slice_fs(_d_w1_fs)

        # u 序列 [T, n_irr_dofs]
        u_seq = np.stack([irr_dict[d][start:end] for d in cfg.irr_dofs], axis=-1).astype(np.float32)

        # 条件向量
        n_freqs_raw = data["natural_freqs"].astype(np.float32) if "natural_freqs" in data else np.zeros(cfg.n_modal, dtype=np.float32)
        n_damps_raw = data["damping_ratios"].astype(np.float32) if "damping_ratios" in data else np.zeros(cfg.n_modal, dtype=np.float32)
        freqs = np.zeros(cfg.n_modal, dtype=np.float32)
        damps = np.zeros(cfg.n_modal, dtype=np.float32)
        n = min(cfg.n_modal, len(n_freqs_raw))
        freqs[:n] = n_freqs_raw[:n]
        damps[:n] = n_damps_raw[:n]
        cond_vec = compute_cond_vecotr(freqs, damps, float(vx_mps), cfg.n_modal)

        # 归一化（使用 train split 统计量）
        if norm_stats is not None:
            y_seq = ((y_seq - norm_stats["y_mean"]) / norm_stats["y_std"]).astype(np.float32)
            u_seq = ((u_seq - norm_stats["u_mean"]) / norm_stats["u_std"]).astype(np.float32)

        x_query = np.linspace(0.0, 1.0, T_use, dtype=np.float32)

        y_list.append(y_seq)
        u_list.append(u_seq)
        phys_x_list.append(phys_x_seq)
        phys_v_list.append(phys_v_seq)
        phys_a_list.append(phys_a_seq)
        phys_z0_list.append(phys_z0_seq)
        phys_z0_w1_list.append(phys_z0_w1_seq)
        phys_z0_w2_list.append(phys_z0_w2_seq)
        phys_z0_w3_list.append(phys_z0_w3_seq)
        phys_z0_w4_list.append(phys_z0_w4_seq)
        xq_list.append(x_query)
        c_list.append(cond_vec.astype(np.float32))
        vx_list.append(np.atleast_1d(np.float32(vx_mps)))
        nf_list.append(freqs)
        dr_list.append(damps)
        src_list.append(str(npz_path))
        seq_len_list.append(T_use)

    if len(y_list) == 0:
        logger.warning(f"[full_seq][{split}] 没有有效样本，跳过。")
        return None

    N = len(y_list)
    T_max = max(seq_len_list)
    n_irr = len(cfg.irr_dofs)

    y_pad = np.zeros((N, T_max, n_sensors), dtype=np.float32)
    u_pad = np.zeros((N, T_max, n_irr), dtype=np.float32)
    phys_x_pad = np.zeros((N, T_max, len(cfg.physics_local_dofs)), dtype=np.float32)
    phys_v_pad = np.zeros((N, T_max, len(cfg.physics_local_dofs)), dtype=np.float32)
    phys_a_pad = np.zeros((N, T_max, len(cfg.physics_local_dofs)), dtype=np.float32)
    phys_z0_pad    = np.zeros((N, T_max, 1), dtype=np.float32)
    phys_z0_w1_pad = np.zeros((N, T_max, 1), dtype=np.float32)
    phys_z0_w2_pad = np.zeros((N, T_max, 1), dtype=np.float32)
    phys_z0_w3_pad = np.zeros((N, T_max, 1), dtype=np.float32)
    phys_z0_w4_pad = np.zeros((N, T_max, 1), dtype=np.float32)
    xq_pad = np.zeros((N, T_max), dtype=np.float32)
    for i in range(N):
        T = seq_len_list[i]
        y_pad[i, :T, :] = y_list[i]
        u_pad[i, :T, :] = u_list[i]
        phys_x_pad[i, :T, :] = phys_x_list[i]
        phys_v_pad[i, :T, :] = phys_v_list[i]
        phys_a_pad[i, :T, :] = phys_a_list[i]
        phys_z0_pad[i, :T, :]    = phys_z0_list[i]
        phys_z0_w1_pad[i, :T, :] = phys_z0_w1_list[i]
        phys_z0_w2_pad[i, :T, :] = phys_z0_w2_list[i]
        phys_z0_w3_pad[i, :T, :] = phys_z0_w3_list[i]
        phys_z0_w4_pad[i, :T, :] = phys_z0_w4_list[i]
        xq_pad[i, :T] = xq_list[i]

    arrays = {
        "y": y_pad,
        "u": u_pad,
        "phys_x": phys_x_pad,
        "phys_v": phys_v_pad,
        "phys_a": phys_a_pad,
        "phys_z0":     phys_z0_pad,
        "phys_z0_w1": phys_z0_w1_pad,
        "phys_z0_w2": phys_z0_w2_pad,
        "phys_z0_w3": phys_z0_w3_pad,
        "phys_z0_w4": phys_z0_w4_pad,
        "x_query": xq_pad,
        "c": np.stack(c_list, axis=0).astype(np.float32),
        "vx": np.stack(vx_list, axis=0).astype(np.float32),
        "natural_freqs": np.stack(nf_list, axis=0).astype(np.float32),
        "damping_ratios": np.stack(dr_list, axis=0).astype(np.float32),
        "seq_lengths": np.asarray(seq_len_list, dtype=np.int32),
        "source_file": np.asarray(src_list, dtype=object),
    }
    logger.info(f"[full_seq][{split}] {N} 个样本，T_max={T_max}")
    return arrays


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 归一化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_norm(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """计算 y 和 u 的均值/标准差（在 train split 上）。"""
    y = arrays["y"]   # [N, T, n_sensors]
    u = arrays["u"]   # [N, L, n_irr]

    y_flat = y.reshape(-1, y.shape[-1])   # [N*T, n_sensors]
    u_flat = u.reshape(-1, u.shape[-1])   # [N*L, n_irr]

    return {
        "y_mean": y_flat.mean(axis=0).astype(np.float32),   # [n_sensors]
        "y_std":  (y_flat.std(axis=0) + 1e-8).astype(np.float32),
        "u_mean": u_flat.mean(axis=0).astype(np.float32),   # [n_irr]
        "u_std":  (u_flat.std(axis=0) + 1e-8).astype(np.float32),
    }


def _apply_norm(
    arrays:     Dict[str, np.ndarray],
    norm_stats: Dict[str, np.ndarray],
) -> None:
    """原地归一化 y 和 u（零均值单位方差）。"""
    arrays["y"] = ((arrays["y"] - norm_stats["y_mean"]) / norm_stats["y_std"]).astype(np.float32)
    arrays["u"] = ((arrays["u"] - norm_stats["u_mean"]) / norm_stats["u_std"]).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_inverse_hdf5_dataset(cfg: InverseDatasetGenConfig) -> Dict:
    """
    生成逆向训练 HDF5 数据集，返回各 split 统计信息字典。
    """
    export_dir = Path(cfg.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # ── 发现 npz 文件 ──────────────────────────────────────────────────────
    use_single = bool(cfg.source_dir.strip())
    if use_single:
        all_npz = sorted(Path(cfg.source_dir).rglob("simulation_result.npz"))
        if not all_npz:
            raise FileNotFoundError(f"在 source_dir 下未找到 simulation_result.npz: {cfg.source_dir}")
        logger.info(f"source_dir 模式：共 {len(all_npz)} 个文件")
        tr_files, va_files, te_files = _split_files(all_npz, cfg.split_ratio, cfg.split_seed)
    else:
        tr_files = sorted(Path(cfg.train_dir).rglob("simulation_result.npz"))
        te_files = sorted(Path(cfg.test_dir).rglob("simulation_result.npz"))
        va_files = te_files
        logger.info(f"train/test_dir 模式：train={len(tr_files)}, test={len(te_files)}")

    # ── 物化 ───────────────────────────────────────────────────────────────
    logger.info("── 物化 train split ──")
    tr_arr = _materialize(tr_files, cfg, "train")
    if tr_arr is None:
        raise RuntimeError("train split 为空，请检查数据目录。")

    logger.info("── 物化 val split ──")
    va_arr = _materialize(va_files, cfg, "val")

    logger.info("── 物化 test split ──")
    te_arr = _materialize(te_files, cfg, "test")

    # ── 归一化 ─────────────────────────────────────────────────────────────
    norm_stats = _compute_norm(tr_arr)
    _apply_norm(tr_arr, norm_stats)
    if va_arr is not None:
        _apply_norm(va_arr, norm_stats)
    if te_arr is not None:
        _apply_norm(te_arr, norm_stats)

    # ── 保存归一化参数 ──────────────────────────────────────────────────────
    np.savez(export_dir / "norm_stats.npz", **norm_stats)
    logger.info(f"归一化参数已保存: {export_dir / 'norm_stats.npz'}")

    # ── HDF5 根级属性 ──────────────────────────────────────────────────────
    root_attrs = {
        "format":               "VTCM_inverse_v1",
        "window_size":          cfg.window_size,
        "stride":               cfg.stride,
        "spatial_len":          cfg.spatial_len,
        "target_freq_hz":       cfg.target_freq,
        "n_modal":              cfg.n_modal,
        "n_sensors":            len(cfg.sensor_local_dofs),
        "n_irr_dofs":           len(cfg.irr_dofs),
        "sensor_local_dofs":    ",".join(str(d) for d in cfg.sensor_local_dofs),
        "physics_local_dofs":   ",".join(str(d) for d in cfg.physics_local_dofs),
        "skip_initial_seconds": cfg.skip_initial_seconds,
    }

    # ── 写入 HDF5 ──────────────────────────────────────────────────────────
    logger.info("── 写入 HDF5 ──")
    _write_hdf5(export_dir / "train.hdf5", tr_arr, root_attrs)
    if te_arr is not None:
        _write_hdf5(export_dir / "test.hdf5", te_arr, root_attrs)
    if va_arr is not None:
        _write_hdf5(export_dir / "validation.hdf5", va_arr, root_attrs)
    elif cfg.write_validation_alias and te_arr is not None:
        _write_hdf5(export_dir / "validation.hdf5", te_arr, root_attrs)

    # ── full_seq（每文件一个样本） ────────────────────────────────────────
    full_seq_info: Dict[str, object] = {}
    if cfg.include_full_sequences:
        logger.info("── 生成 full_seq 数据集 ──")
        tr_full = _materialize_full_sequences(tr_files, cfg, "train", norm_stats=norm_stats)
        va_full = _materialize_full_sequences(va_files, cfg, "validation", norm_stats=norm_stats)
        te_full = _materialize_full_sequences(te_files, cfg, "test", norm_stats=norm_stats)

        full_attrs = dict(root_attrs)
        full_attrs["format"] = "VTCM_inverse_full_seq_v1"

        if tr_full is not None:
            _write_hdf5(export_dir / "train_full_seq.hdf5", tr_full, full_attrs)
        if va_full is not None:
            _write_hdf5(export_dir / "validation_full_seq.hdf5", va_full, full_attrs)
        elif cfg.write_validation_alias and te_full is not None:
            _write_hdf5(export_dir / "validation_full_seq.hdf5", te_full, full_attrs)
        if te_full is not None:
            _write_hdf5(export_dir / "test_full_seq.hdf5", te_full, full_attrs)

        full_seq_info = {
            "train_full_seq_hdf5": str(export_dir / "train_full_seq.hdf5"),
            "val_full_seq_hdf5": str(export_dir / "validation_full_seq.hdf5"),
            "test_full_seq_hdf5": str(export_dir / "test_full_seq.hdf5"),
            "train_full_seq_samples": int(tr_full["y"].shape[0]) if tr_full is not None else 0,
            "val_full_seq_samples": int(va_full["y"].shape[0]) if va_full is not None else 0,
            "test_full_seq_samples": int(te_full["y"].shape[0]) if te_full is not None else 0,
            "train_full_seq_T_max": int(tr_full["y"].shape[1]) if tr_full is not None else 0,
        }

    # ── 写 meta.json ───────────────────────────────────────────────────────
    meta: Dict = {
        "format":               "VTCM_inverse_v1",
        "source_dir":           cfg.source_dir,
        "train_dir":            cfg.train_dir,
        "test_dir":             cfg.test_dir,
        "export_dir":           str(export_dir),
        "split_ratio":          list(cfg.split_ratio),
        "split_seed":           cfg.split_seed,
        "window_size":          cfg.window_size,
        "stride":               cfg.stride,
        "spatial_len":          cfg.spatial_len,
        "target_freq_hz":       cfg.target_freq,
        "n_modal":              cfg.n_modal,
        "sensor_local_dofs":    list(cfg.sensor_local_dofs),
        "physics_local_dofs":   list(cfg.physics_local_dofs),
        "irr_dofs":             list(cfg.irr_dofs),
        "skip_initial_seconds": cfg.skip_initial_seconds,
        "train_samples":  int(tr_arr["y"].shape[0]),
        "val_samples":    int(va_arr["y"].shape[0]) if va_arr is not None else 0,
        "test_samples":   int(te_arr["y"].shape[0]) if te_arr is not None else 0,
        "y_shape":  list(tr_arr["y"].shape[1:]),    # [T, n_sensors]
        "u_shape":  list(tr_arr["u"].shape[1:]),    # [L, n_irr_dofs]
        "phys_x_shape": list(tr_arr["phys_x"].shape[1:]),  # [T, 7]
        "phys_v_shape": list(tr_arr["phys_v"].shape[1:]),  # [T, 7]
        "phys_a_shape": list(tr_arr["phys_a"].shape[1:]),  # [T, 7]
        "phys_z0_shape": list(tr_arr["phys_z0"].shape[1:]),# [T, 1]
        "c_shape":  list(tr_arr["c"].shape[1:]),    # [n_cond]
    }
    if full_seq_info:
        meta["full_seq"] = full_seq_info
    (export_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"meta.json 已保存: {export_dir / 'meta.json'}")
    return meta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="InverseDatasetGen: 生成逆向模型 HDF5 数据集")
    p.add_argument("--source_dir",  type=str, default=str(_ROOT / "results" / "pino_train_dataset"),
                   help="仿真结果根目录（按 split_ratio 切分 train/val/test）")
    p.add_argument("--train_dir",   type=str, default=str(_ROOT / "results" / "pino_train_dataset"),
                   help="若指定，与 --test_dir 一起使用，忽略 source_dir")
    p.add_argument("--test_dir",    type=str, default=str(_ROOT / "results" / "pino_test_dataset"),
                   help="测试集目录（与 --train_dir 配合使用）")
    p.add_argument("--export_dir",  type=str, default=str(_ROOT / "datasets" / "VTCM_inverse"),
                   help="HDF5 导出目录")
    p.add_argument("--split_ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--split_seed",  type=int,   default=42)
    p.add_argument("--window_size", type=int,   default=256)
    p.add_argument("--stride",      type=int,   default=128)
    p.add_argument("--spatial_len", type=int,   default=256)
    p.add_argument("--target_freq", type=int,   default=10_000)
    p.add_argument("--sim_freq",    type=int,   default=10_000)
    p.add_argument("--n_modal",     type=int,   default=8)
    p.add_argument("--skip_seconds",type=float, default=2.0)
    p.add_argument("--disable_full_seq", action="store_true",
                   help="关闭 full_seq 导出（默认开启）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = InverseDatasetGenConfig(
        source_dir            = args.source_dir,
        train_dir             = args.train_dir,
        test_dir              = args.test_dir,
        export_dir            = args.export_dir,
        split_ratio           = tuple(args.split_ratio),
        split_seed            = args.split_seed,
        window_size           = args.window_size,
        stride                = args.stride,
        spatial_len           = args.spatial_len,
        target_freq           = args.target_freq,
        sim_freq              = args.sim_freq,
        n_modal               = args.n_modal,
        skip_initial_seconds  = args.skip_seconds,
        include_full_sequences= not args.disable_full_seq,
    )

    meta = build_inverse_hdf5_dataset(cfg)

    logger.info("\n=== 数据集生成完成 ===")
    logger.info(f"  输出目录:   {meta['export_dir']}")
    logger.info(f"  train:      {meta['train_samples']} 窗口")
    logger.info(f"  val:        {meta['val_samples']} 窗口")
    logger.info(f"  test:       {meta['test_samples']} 窗口")
    logger.info(f"  y_shape:    {meta['y_shape']}   [T, n_sensors]")
    logger.info(f"  u_shape:    {meta['u_shape']}   [L, n_irr_dofs]")
    logger.info(f"  phys_x:     {meta['phys_x_shape']} [T, 7]")
    logger.info(f"  phys_v:     {meta['phys_v_shape']} [T, 7]")
    logger.info(f"  phys_a:     {meta['phys_a_shape']} [T, 7]")
    logger.info(f"  phys_z0:    {meta['phys_z0_shape']} [T, 1]")
    logger.info(f"  c_shape:    {meta['c_shape']}   [n_cond]")


if __name__ == "__main__":
    main()
