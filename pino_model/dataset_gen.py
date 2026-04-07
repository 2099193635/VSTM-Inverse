"""
Dataset generation module for PINO training.
Handles loading, preprocessing, and normalization of simulation data.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dataset_generator import WindowConfig, build_dataloader


def _load_run_metadata(npz_path: Path) -> Dict[str, Any]:
    """Load simulation metadata from argparse_params.json if available."""
    meta = {
        "vehicle_type": "高速客车",
        "param_profile_dir": "configs/standard",
        "vx_set": 215.0,
        "fastener_type": "Standard_KV",
        "g": 9.81,
    }
    json_path = npz_path.parent / "argparse_params.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            meta.update({k: raw[k] for k in meta.keys() if k in raw and raw[k] not in (None, "")})
    return meta


def _target_indices(target_object: str) -> np.ndarray:
    """Get DOF indices for a target object."""
    target_object = target_object.lower()
    if target_object == "carbody":
        return np.array([1], dtype=np.int64)
    if target_object == "bogie":
        return np.array([6, 11], dtype=np.int64)
    if target_object == "wheelset":
        return np.array([16, 21, 26, 31], dtype=np.int64)
    if target_object == "all_vehicle":
        return np.array([1, 6, 11, 16, 21, 26, 31], dtype=np.int64)
    if target_object in {"vehicle_full", "full_vehicle", "vehicle35"}:
        return np.arange(35, dtype=np.int64)
    raise ValueError(f"Unsupported target object: {target_object}")


def _parse_components(components: str) -> List[str]:
    """Parse comma-separated component names."""
    valid = {"disp": "X", "vel": "V", "acc": "A"}
    comps = [c.strip().lower() for c in components.split(",") if c.strip()]
    if not comps:
        raise ValueError("components cannot be empty")
    for c in comps:
        if c not in valid:
            raise ValueError(f"Unsupported component: {c}, expected one of {list(valid.keys())}")
    return comps


def _load_aux_series(
    data, key: str, t_len: int, width: Optional[int] = None
) -> Optional[np.ndarray]:
    """Load auxiliary array from npz data with validation."""
    if key not in data:
        return None
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return None
    arr = arr[:t_len]
    if arr.shape[0] != t_len:
        return None
    if width is not None and arr.shape[1] != width:
        return None
    return arr.astype(np.float32)


def _build_z_from_npz(
    data,
    target_object: str,
    components: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]], List[int]]:
    """Build state matrix from NPZ data for selected target object and components."""
    key_map = {"disp": "X", "vel": "V", "acc": "A"}
    idx = _target_indices(target_object)

    z_parts: List[np.ndarray] = []
    seg: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for c in components:
        k = key_map[c]
        if k not in data:
            raise KeyError(f"{k} not found in NPZ while component '{c}' is requested")
        arr = np.asarray(data[k], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected {k} with shape [T, C], got {arr.shape}")
        if arr.shape[1] < idx.max() + 1:
            raise ValueError(
                f"{k} has only {arr.shape[1]} channels, target needs index {idx.max()}"
            )
        sel = arr[:, idx]
        z_parts.append(sel)
        seg[c] = (cursor, cursor + sel.shape[1])
        cursor += sel.shape[1]

    z = np.concatenate(z_parts, axis=1).astype(np.float32)
    return z, seg, idx.tolist()


def load_records_from_npz(
    npz_path: Path,
    target_object: str,
    components: Sequence[str],
    u_keys: Sequence[str] = (
        "Irre_bz_L_ref",
        "Irre_bz_R_ref",
        "Irre_by_L_ref",
        "Irre_by_R_ref",
    ),
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Tuple[int, int]], List[int]]:
    """Load simulation records from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    z, component_segments, selected_indices = _build_z_from_npz(data, target_object, components)

    u_list = []
    for k in u_keys:
        if k in data:
            col = np.asarray(data[k], dtype=np.float32).reshape(-1, 1)
            if len(col) != len(z):
                continue
            u_list.append(col)

    if len(u_list) == 0:
        u = np.zeros((len(z), 1), dtype=np.float32)
    else:
        u = np.concatenate(u_list, axis=1).astype(np.float32)

    meta = _load_run_metadata(npz_path)
    v_kmh_npz = None
    if "v_kmh" in data:
        v_kmh_raw = np.asarray(data["v_kmh"]).reshape(-1)
        if v_kmh_raw.size > 0:
            v_kmh_npz = float(v_kmh_raw[0])
    c = np.array(
        [float(v_kmh_npz if v_kmh_npz is not None else meta.get("vx_set", 0.0))],
        dtype=np.float32,
    )
    record: Dict[str, np.ndarray] = {"z": z, "u": u, "c": c}

    if "spatial_s" in data:
        spatial_s = np.asarray(data["spatial_s"], dtype=np.float32).reshape(-1)
        if spatial_s.shape[0] >= len(z):
            record["spatial_s"] = spatial_s[: len(z)]

    wr_vertical = _load_aux_series(data, "TotalVerticalForce", len(z), width=8)
    wr_lateral = _load_aux_series(data, "TotalLateralForce", len(z), width=8)
    wr_vertical_p2 = _load_aux_series(data, "TotalVerticalForce_Point2", len(z), width=8)
    wr_lateral_p2 = _load_aux_series(data, "TotalLateralForce_Point2", len(z), width=8)

    if wr_vertical is not None:
        record["wr_force_vertical"] = wr_vertical
    if wr_lateral is not None:
        record["wr_force_lateral"] = wr_lateral
    if wr_vertical_p2 is not None:
        record["wr_force_vertical_p2"] = wr_vertical_p2
    if wr_lateral_p2 is not None:
        record["wr_force_lateral_p2"] = wr_lateral_p2

    return [record], component_segments, selected_indices


def find_npz_files(directory: Path, pattern: str = "*.npz") -> List[Path]:
    """
    Recursively find all NPZ files in a directory.
    
    Args:
        directory: Root directory to search
        pattern: File pattern to match (default: "*.npz")
    
    Returns:
        List of paths to NPZ files found
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    
    npz_files = list(directory.rglob(pattern))
    npz_files.sort()
    
    return npz_files


def load_records_from_directory(
    directory: Path,
    target_object: str,
    components: Sequence[str],
    u_keys: Sequence[str] = (
        "Irre_bz_L_ref",
        "Irre_bz_R_ref",
        "Irre_by_L_ref",
        "Irre_by_R_ref",
    ),
    pattern: str = "*spatial.npz",
    verbose: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Tuple[int, int]], List[int]]:
    """
    Load records from all matching NPZ files in a directory.
    
    Args:
        directory: Directory containing NPZ files
        target_object: Target object ("carbody", "bogie", "wheelset", "vehicle_full", etc.)
        components: Sequence of components ("disp", "vel", "acc")
        u_keys: Keys for disturbance data
        pattern: File pattern to match (default: "*spatial.npz")
        verbose: Print progress information
    
    Returns:
        (records, component_segments, selected_indices)
        - records: List of loaded record dictionaries
        - component_segments: Mapping of component names to column ranges
        - selected_indices: DOF indices of selected target object
    """
    directory = Path(directory)
    npz_files = find_npz_files(directory, pattern)
    
    if not npz_files:
        raise FileNotFoundError(
            f"No NPZ files matching pattern '{pattern}' found in {directory}"
        )
    
    if verbose:
        print(f"[Dataset] 找到 {len(npz_files)} 个 NPZ 文件: {pattern}")
    
    all_records: List[Dict[str, np.ndarray]] = []
    component_segments: Dict[str, Tuple[int, int]] = {}
    selected_indices: List[int] = []
    
    for i, npz_path in enumerate(npz_files):
        try:
            records, comp_segs, idx = load_records_from_npz(
                npz_path,
                target_object=target_object,
                components=components,
                u_keys=u_keys,
            )
            all_records.extend(records)
            
            # Store component segments and indices (should be same for all files)
            if not component_segments:
                component_segments = comp_segs
                selected_indices = idx
            
            if verbose:
                print(f"  [{i+1}/{len(npz_files)}] 已加载: {npz_path.name} ({len(records)} 条记录)")
        
        except Exception as e:
            if verbose:
                print(f"  [警告] 跳过文件 {npz_path.name}: {str(e)}")
            continue
    
    if not all_records:
        raise RuntimeError(f"无法从 {directory} 加载任何有效的记录")
    
    if verbose:
        print(f"[Dataset] 总共加载了 {len(all_records)} 条记录")
    
    return all_records, component_segments, selected_indices


def load_records_from_file_list(
    file_list: List[Path],
    target_object: str,
    components: Sequence[str],
    u_keys: Sequence[str] = (
        "Irre_bz_L_ref",
        "Irre_bz_R_ref",
        "Irre_by_L_ref",
        "Irre_by_R_ref",
    ),
    verbose: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Tuple[int, int]], List[int]]:
    """
    Load records from a list of NPZ files.
    
    Args:
        file_list: List of paths to NPZ files
        target_object: Target object specification
        components: Components to load
        u_keys: Keys for disturbance data
        verbose: Print progress information
    
    Returns:
        (records, component_segments, selected_indices)
    """
    if not file_list:
        raise ValueError("file_list cannot be empty")
    
    file_list = [Path(f) for f in file_list]
    
    if verbose:
        print(f"[Dataset] 加载 {len(file_list)} 个文件")
    
    all_records: List[Dict[str, np.ndarray]] = []
    component_segments: Dict[str, Tuple[int, int]] = {}
    selected_indices: List[int] = []
    
    for i, npz_path in enumerate(file_list):
        try:
            records, comp_segs, idx = load_records_from_npz(
                Path(npz_path),
                target_object=target_object,
                components=components,
                u_keys=u_keys,
            )
            all_records.extend(records)
            
            if not component_segments:
                component_segments = comp_segs
                selected_indices = idx
            
            if verbose:
                print(f"  [{i+1}/{len(file_list)}] 已加载: {npz_path.name}")
        
        except Exception as e:
            if verbose:
                print(f"  [警告] 跳过文件 {npz_path.name}: {str(e)}")
            continue
    
    if not all_records:
        raise RuntimeError(f"无法从文件列表加载任何有效的记录")
    
    if verbose:
        print(f"[Dataset] 总共加载了 {len(all_records)} 条记录")
    
    return all_records, component_segments, selected_indices


def build_demo_records(
    n_records: int = 16, t_len: int = 2000, cz: int = 8, cu: int = 4
) -> List[Dict[str, np.ndarray]]:
    """Generate synthetic demo records for testing."""
    rng = np.random.default_rng(42)
    records: List[Dict[str, np.ndarray]] = []

    for _ in range(n_records):
        u = rng.standard_normal((t_len, cu)).astype(np.float32) * 0.3
        z = np.zeros((t_len, cz), dtype=np.float32)

        A = 0.96 * np.eye(cz, dtype=np.float32)
        B = rng.standard_normal((cz, cu)).astype(np.float32) * 0.05

        for t in range(1, t_len):
            z[t] = (
                A @ z[t - 1]
                + B @ u[t - 1]
                + 0.01 * rng.standard_normal(cz).astype(np.float32)
            )

        c = np.array([rng.uniform(40.0, 120.0)], dtype=np.float32)  # e.g., speed
        records.append({"z": z, "u": u, "c": c})

    return records


def split_records_temporal(
    records: List[Dict[str, np.ndarray]],
    test_ratio: float,
    min_seq_len: int,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Split records into temporal train and test sets."""
    train_records: List[Dict[str, np.ndarray]] = []
    test_records: List[Dict[str, np.ndarray]] = []

    for rec in records:
        z = rec["z"]
        u = rec.get("u")
        y = rec.get("y")
        c = rec.get("c")
        t_len = len(z)

        split = int((1.0 - test_ratio) * t_len)
        split = max(min_seq_len, min(split, t_len - min_seq_len))

        tr = {"z": z[:split], "c": c}
        te = {"z": z[split:], "c": c}
        if u is not None:
            tr["u"] = u[:split]
            te["u"] = u[split:]
        if y is not None:
            tr["y"] = y[:split]
            te["y"] = y[split:]

        if len(tr["z"]) >= min_seq_len:
            train_records.append(tr)
        if len(te["z"]) >= min_seq_len:
            test_records.append(te)

    return train_records, test_records


def compute_z_norm_stats(
    records: Sequence[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalization statistics from training records."""
    all_z = np.concatenate([np.asarray(r["z"], dtype=np.float32) for r in records], axis=0)
    mu = np.mean(all_z, axis=0, keepdims=True)
    sigma = np.std(all_z, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_z_norm(
    records: Sequence[Dict[str, np.ndarray]], mu: np.ndarray, sigma: np.ndarray
) -> List[Dict[str, np.ndarray]]:
    """Apply z-score normalization to records."""
    out: List[Dict[str, np.ndarray]] = []
    for r in records:
        nr = dict(r)
        nr["z"] = (np.asarray(r["z"], dtype=np.float32) - mu) / sigma
        out.append(nr)
    return out


def prepare_dataset(
    npz_path: Optional[Path] = None,
    target_object: str = "vehicle_full",
    components: str = "disp,vel,acc",
    window_size: int = 256,
    stride: int = 64,
    pred_horizon: int = 1,
    batch_size: int = 8,
    test_ratio: float = 0.2,
    u_keys: Sequence[str] = (
        "Irre_bz_L_ref",
        "Irre_bz_R_ref",
        "Irre_by_L_ref",
        "Irre_by_R_ref",
    ),
    demo: bool = False,
    demo_records: int = 16,
    demo_length: int = 2000,
    dataset_dir: Optional[Path] = None,
    npz_dir: Optional[Path] = None,
    file_list: Optional[List[Path]] = None,
    npz_pattern: str = "*spatial.npz",
) -> Tuple[Any, Any, Dict[str, Tuple[int, int]], List[int], np.ndarray, np.ndarray, float, List[Dict[str, np.ndarray]]]:


    components_list = _parse_components(components)
    component_segments: Dict[str, Tuple[int, int]] = {}
    selected_indices: List[int] = []

    if demo:
        # Demo mode uses synthetic z
        demo_dim = 35
        base = build_demo_records(n_records=demo_records, t_len=demo_length, cz=demo_dim, cu=4)
        records = []
        idx = _target_indices(target_object)
        for rec in base:
            x = rec["z"]
            v = np.gradient(x, axis=0)
            a = np.gradient(v, axis=0)
            z_parts = []
            cursor = 0
            for c in components_list:
                arr = {"disp": x, "vel": v, "acc": a}[c][:, idx]
                z_parts.append(arr)
                component_segments[c] = (cursor, cursor + arr.shape[1])
                cursor += arr.shape[1]
            z = np.concatenate(z_parts, axis=1).astype(np.float32)
            records.append({"z": z, "u": rec["u"], "c": rec["c"]})
        selected_indices = idx.tolist()
        ds = 0.25
    
    elif npz_dir is not None:
        # Load from directory (recursive)
        print(f"[Dataset] 从目录加载数据: {npz_dir}")
        records, component_segments, selected_indices = load_records_from_directory(
            npz_dir,
            target_object=target_object,
            components=components_list,
            u_keys=u_keys,
            pattern=npz_pattern,
            verbose=True,
        )
        ds = 0.25
        # Auto-detect spatial step if available
        rec0 = records[0] if records else None
        if rec0 is not None and "spatial_s" in rec0:
            s_arr = np.asarray(rec0["spatial_s"], dtype=np.float32).reshape(-1)
            if s_arr.size > 1:
                ds_auto = float(np.median(np.diff(s_arr)))
                if ds_auto > 0:
                    ds = ds_auto
    
    elif file_list is not None:
        # Load from file list
        print(f"[Dataset] 从文件列表加载数据 ({len(file_list)} 个文件)")
        records, component_segments, selected_indices = load_records_from_file_list(
            file_list,
            target_object=target_object,
            components=components_list,
            u_keys=u_keys,
            verbose=True,
        )
        ds = 0.25
        # Auto-detect spatial step if available
        rec0 = records[0] if records else None
        if rec0 is not None and "spatial_s" in rec0:
            s_arr = np.asarray(rec0["spatial_s"], dtype=np.float32).reshape(-1)
            if s_arr.size > 1:
                ds_auto = float(np.median(np.diff(s_arr)))
                if ds_auto > 0:
                    ds = ds_auto
    
    elif npz_path is not None:
        # Load from single file
        records, component_segments, selected_indices = load_records_from_npz(
            npz_path,
            target_object=target_object,
            components=components_list,
            u_keys=u_keys,
        )
        ds = 0.25
        # Auto-detect spatial step if available
        rec0 = records[0] if records else None
        if rec0 is not None and "spatial_s" in rec0:
            s_arr = np.asarray(rec0["spatial_s"], dtype=np.float32).reshape(-1)
            if s_arr.size > 1:
                ds_auto = float(np.median(np.diff(s_arr)))
                if ds_auto > 0:
                    ds = ds_auto
    
    else:
        raise ValueError(
            "Must provide one of: npz_path, npz_dir, file_list, or demo=True"
        )

    min_len = window_size + pred_horizon + 1
    train_records, test_records = split_records_temporal(records, test_ratio, min_len)
    if len(train_records) == 0 or len(test_records) == 0:
        raise RuntimeError(
            "Insufficient data after temporal split. Try smaller window size or lower test ratio."
        )

    z_mu, z_sigma = compute_z_norm_stats(train_records)
    train_records = apply_z_norm(train_records, z_mu, z_sigma)
    test_records = apply_z_norm(test_records, z_mu, z_sigma)

    # Save normalization parameters
    if dataset_dir:
        dataset_dir = Path(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            dataset_dir / "norm_stats.npz",
            z_mu=z_mu,
            z_sigma=z_sigma,
            ds=np.array([ds]),
        )
        print(f"[Dataset] 已保存归一化参数到 {dataset_dir / 'norm_stats.npz'}")

    train_loader = build_dataloader(
        records=train_records,
        config=WindowConfig(window_size=window_size, stride=stride, pred_horizon=pred_horizon),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = build_dataloader(
        records=test_records,
        config=WindowConfig(window_size=window_size, stride=stride, pred_horizon=pred_horizon),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, component_segments, selected_indices, z_mu, z_sigma, ds, test_records


if __name__ == "__main__":
    # Example usage
    prepare_dataset(npz_path=''
    

    )
    print("Dataset generation module loaded successfully.")
