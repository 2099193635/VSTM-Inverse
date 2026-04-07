from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

try:
    from scipy.signal import butter, filtfilt, welch

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


@dataclass
class Config:
    # 推荐输入：结果根目录（非 files 目录）
    # 例如：results/default_project/高速客车-外部导入-vehicle-standard-20260322_065703
    result_root_dir: Path = Path(
        r"results/default_project/高速客车-外部导入-vehicle-standard-20260322_065703"
    )

    # 兼容保留：当 result_root_dir 为 None 时，使用这两个显式路径
    sim_npz_path: Optional[Path] = None
    sim_json_path: Optional[Path] = None
    measured_csv_path: Path = Path(
        r"preprocessing/动检数据/呼局/20210416/处理后/动检上行20210416-238-363.processed.csv"
    )

    sim_channel: str = "wheelset_1_axle_z"
    measured_channel: str = "measured_left_irr_mm"
    include_static_irr: bool = True
    # 静态对比位置修正（单位 m，正值表示沿里程正方向取样）。
    # None: 自动按 sim_channel 与车辆几何(Lc/Lt)推断；
    # 例如前后转向架均值可设为 (-9.0, 9.0) 或按轴位设为四点均值。
    static_compare_offsets_m: Optional[Tuple[float, ...]] = (-9.0, 9.0)

    lead_time_s: float = 2.0
    search_window_m: float = 400.0
    target_spacing_m: float = 0.25
    lowpass_cutoff_hz: float = 25.0
    max_start_candidates: int = 400

    figure_width_in: float = 7.2
    figure_height_in: float = 5.6
    figure_dpi: int = 140
    export_dpi: int = 300
    export_png: bool = True
    export_pdf: bool = True
    output_dir: Optional[Path] = None
    output_stem: str = "dynamic_irregularity_comparison"
    # 分频段误差对比（Hz）
    freq_bands_hz: Tuple[Tuple[float, float], ...] = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0))

    def __post_init__(self) -> None:
        # 优先根据结果根目录自动推断读取与保存路径
        if self.result_root_dir is not None:
            root = Path(self.result_root_dir)
            if root.name.lower() == "files":
                root = root.parent
            if self.sim_npz_path is None:
                self.sim_npz_path = root / "files" / "simulation_result.npz"
            if self.sim_json_path is None:
                self.sim_json_path = root / "files" / "argparse_params.json"
            if self.output_dir is None:
                self.output_dir = root / "figures"

        # 如果未设置 result_root_dir，则回退到显式路径
        if self.sim_npz_path is None or self.sim_json_path is None:
            raise ValueError("sim_npz_path / sim_json_path 未设置，请提供 result_root_dir 或显式文件路径。")

        if self.output_dir is None:
            self.output_dir = Path("results/figures")


def setup_publication_style() -> None:
    """Set matplotlib style for journal-grade figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 2.5,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.30,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def compute_metrics(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return correlation, RMSE, and NRMSE after mean removal."""
    n = min(len(x), len(y))
    x0 = np.asarray(x[:n], dtype=float) - np.mean(x[:n])
    y0 = np.asarray(y[:n], dtype=float) - np.mean(y[:n])
    corr = float(np.corrcoef(x0, y0)[0, 1])
    rmse = float(np.sqrt(np.mean((x0 - y0) ** 2)))
    nrmse = float(rmse / (np.std(y0) + 1e-12))
    return corr, rmse, nrmse


def lowpass_fft(x: np.ndarray, dt: float, fc: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    freq = np.fft.rfftfreq(len(x), d=dt)
    xf = np.fft.rfft(x)
    xf[freq > fc] = 0.0
    return np.fft.irfft(xf, n=len(x))


def lowpass_filter(x: np.ndarray, fs: float, dt: float, fc: float, order: int = 4) -> np.ndarray:
    if SCIPY_AVAILABLE:
        wn = fc / (0.5 * fs)
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = butter(order, wn, btype="low")
        return filtfilt(b, a, x)
    return lowpass_fft(x, dt, fc)


def fft_power(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(x, dtype=float) - np.mean(x)
    freq = np.fft.rfftfreq(len(x0), d=dt)
    xf = np.fft.rfft(x0)
    power = (np.abs(xf) ** 2) / max(len(x0), 1)
    return freq, power


def bandpass_fft(x: np.ndarray, dt: float, f_low: float, f_high: float) -> np.ndarray:
    """Simple FFT band-pass (inclusive) for metric comparison."""
    x = np.asarray(x, dtype=float)
    x0 = x - np.mean(x)
    freq = np.fft.rfftfreq(len(x0), d=dt)
    xf = np.fft.rfft(x0)
    keep = (freq >= max(0.0, float(f_low))) & (freq <= max(float(f_high), float(f_low)))
    xf[~keep] = 0.0
    return np.fft.irfft(xf, n=len(x0))


def build_improvement_tables(best: dict, cfg: Config, analysis: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall and band-wise comparison tables to highlight dynamic-vs-static improvement."""
    rows = []

    def _safe_improve(base: float, new: float) -> float:
        if not np.isfinite(base) or abs(base) < 1e-12:
            return np.nan
        return float((base - new) / base * 100.0)

    nrmse_dyn_raw = float(best["nrmse_raw"])
    corr_dyn_raw = float(best["corr_raw"])
    nrmse_dyn_lp = float(best["nrmse_lp"])
    corr_dyn_lp = float(best["corr_lp"])
    nrmse_sta_raw = float(best.get("nrmse_static_raw", np.nan))
    corr_sta_raw = float(best.get("corr_static_raw", np.nan))
    nrmse_sta_lp = float(best.get("nrmse_static_lp", np.nan))
    corr_sta_lp = float(best.get("corr_static_lp", np.nan))

    rows.append(
        {
            "case": "raw",
            "corr_dynamic": corr_dyn_raw,
            "corr_static": corr_sta_raw,
            "delta_corr_dyn_minus_static": corr_dyn_raw - corr_sta_raw if np.isfinite(corr_sta_raw) else np.nan,
            "nrmse_dynamic": nrmse_dyn_raw,
            "nrmse_static": nrmse_sta_raw,
            "nrmse_improvement_pct": _safe_improve(nrmse_sta_raw, nrmse_dyn_raw),
        }
    )
    rows.append(
        {
            "case": "lowpass",
            "corr_dynamic": corr_dyn_lp,
            "corr_static": corr_sta_lp,
            "delta_corr_dyn_minus_static": corr_dyn_lp - corr_sta_lp if np.isfinite(corr_sta_lp) else np.nan,
            "nrmse_dynamic": nrmse_dyn_lp,
            "nrmse_static": nrmse_sta_lp,
            "nrmse_improvement_pct": _safe_improve(nrmse_sta_lp, nrmse_dyn_lp),
        }
    )
    summary_df = pd.DataFrame(rows)

    bands = []
    y = np.asarray(best["measured_raw"], dtype=float)
    xd = np.asarray(best["sim_raw"], dtype=float)
    xs = best.get("static_raw")
    xs_arr = np.asarray(xs, dtype=float) if xs is not None else None
    dt = float(analysis["dt_compare_s"])

    for f1, f2 in cfg.freq_bands_hz:
        yb = bandpass_fft(y, dt, f1, f2)
        xdb = bandpass_fft(xd, dt, f1, f2)
        corr_d, rmse_d, nrmse_d = compute_metrics(xdb, yb)

        if xs_arr is not None:
            xsb = bandpass_fft(xs_arr, dt, f1, f2)
            corr_s, rmse_s, nrmse_s = compute_metrics(xsb, yb)
            delta_corr = corr_d - corr_s
            imp_pct = _safe_improve(nrmse_s, nrmse_d)
        else:
            corr_s, rmse_s, nrmse_s = np.nan, np.nan, np.nan
            delta_corr = np.nan
            imp_pct = np.nan

        bands.append(
            {
                "band_hz": f"{f1:.1f}-{f2:.1f}",
                "corr_dynamic": corr_d,
                "corr_static": corr_s,
                "delta_corr_dyn_minus_static": delta_corr,
                "rmse_dynamic_m": rmse_d,
                "rmse_static_m": rmse_s,
                "nrmse_dynamic": nrmse_d,
                "nrmse_static": nrmse_s,
                "nrmse_improvement_pct": imp_pct,
            }
        )
    band_df = pd.DataFrame(bands)
    return summary_df, band_df


def make_improvement_figure(summary_df: pd.DataFrame, band_df: pd.DataFrame, cfg: Config) -> plt.Figure:
    """Create one concise figure emphasizing dynamic-vs-static improvement."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(cfg.figure_width_in * 1.3, cfg.figure_height_in * 0.52), dpi=cfg.figure_dpi)

    s = summary_df.set_index("case")
    labels = ["Raw", "Low-pass"]
    x = np.arange(len(labels))
    w = 0.36

    dyn_vals = [float(s.loc["raw", "nrmse_dynamic"]), float(s.loc["lowpass", "nrmse_dynamic"])]
    sta_raw = float(s.loc["raw", "nrmse_static"])
    sta_lp = float(s.loc["lowpass", "nrmse_static"])
    sta_vals = [sta_raw if np.isfinite(sta_raw) else 0.0, sta_lp if np.isfinite(sta_lp) else 0.0]

    ax1.bar(x - w / 2, dyn_vals, width=w, color="#1f4e79", label="Dynamic sim")
    ax1.bar(x + w / 2, sta_vals, width=w, color="#4d4d4d", label="Static irr")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("NRMSE")
    ax1.set_title("Overall error comparison", loc="left")
    ax1.grid(True, axis="y", alpha=0.35)
    ax1.legend(frameon=False, loc="upper right")

    imp_vals = np.nan_to_num(band_df["nrmse_improvement_pct"].to_numpy(dtype=float), nan=0.0)
    ax2.bar(np.arange(len(band_df)), imp_vals, color="#1f4e79")
    ax2.axhline(0.0, color="#444444", lw=0.8)
    ax2.set_xticks(np.arange(len(band_df)))
    ax2.set_xticklabels(band_df["band_hz"].tolist())
    ax2.set_ylabel("NRMSE improvement (%)")
    ax2.set_title("Band-wise improvement", loc="left")
    ax2.grid(True, axis="y", alpha=0.35)

    txt = []
    for _, row in summary_df.iterrows():
        d_corr = row["delta_corr_dyn_minus_static"]
        d_imp = row["nrmse_improvement_pct"]
        txt.append(
            f"{row['case']}: Δr={d_corr:.3f}, NRMSE↓={d_imp:.1f}%"
            if np.isfinite(d_corr) and np.isfinite(d_imp)
            else f"{row['case']}: Δr=N/A, NRMSE↓=N/A"
        )
    fig.text(0.995, 0.02, "\n".join(txt), ha="right", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    return fig


def build_sim_channels(x: np.ndarray, params_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build simulation candidate channels using English channel keys."""
    z_cols = {
        "wheelset_1_axle_z": 16,
        "wheelset_2_axle_z": 21,
        "wheelset_3_axle_z": 26,
        "wheelset_4_axle_z": 31,
    }
    roll_cols = {
        "wheelset_1_roll": 17,
        "wheelset_2_roll": 22,
        "wheelset_3_roll": 27,
        "wheelset_4_roll": 32,
    }

    channels = {name: np.asarray(x[:, col], dtype=float) for name, col in z_cols.items()}

    z1 = channels["wheelset_1_axle_z"]
    z2 = channels["wheelset_2_axle_z"]
    z3 = channels["wheelset_3_axle_z"]
    z4 = channels["wheelset_4_axle_z"]

    # 便于直接对比“前后转向架均值 / 全轴均值”动态不平顺
    channels["bogie_front_axle_z_mean"] = 0.5 * (z1 + z2)
    channels["bogie_rear_axle_z_mean"] = 0.5 * (z3 + z4)
    channels["bogie_pair_axle_z_mean"] = 0.25 * (z1 + z2 + z3 + z4)
    channels["wheelset_all_axle_z_mean"] = channels["bogie_pair_axle_z_mean"]

    dw = None
    try:
        if "veh_1" in params_df.columns:
            veh_cfg = params_df["veh_1"].iloc[0]
            if isinstance(veh_cfg, dict) and "dw" in veh_cfg:
                dw = float(veh_cfg["dw"])
    except Exception:
        dw = None

    if dw is not None:
        for i in range(1, 5):
            z_key = f"wheelset_{i}_axle_z"
            r_key = f"wheelset_{i}_roll"
            z_val = np.asarray(x[:, z_cols[z_key]], dtype=float)
            r_val = np.asarray(x[:, roll_cols[r_key]], dtype=float)
            channels[f"wheelset_{i}_left_equiv_z"] = z_val + dw * r_val
            channels[f"wheelset_{i}_right_equiv_z"] = z_val - dw * r_val

    return channels


def build_measured_channels(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build measured candidate channels using English keys."""
    mileage_km = pd.to_numeric(df["里程"], errors="coerce").to_numpy(dtype=float)
    left_irr_mm = pd.to_numeric(df["左高低"], errors="coerce").to_numpy(dtype=float)
    right_irr_mm = pd.to_numeric(df["右高低"], errors="coerce").to_numpy(dtype=float)

    channels = {
        "measured_left_irr_mm": left_irr_mm,
        "measured_right_irr_mm": right_irr_mm,
    }
    return mileage_km, channels


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _parse_external_file_map(params_df: pd.DataFrame) -> Dict[str, Path]:
    """Parse external_files in argparse params and return {VL/VR/LL/LR: absolute_path}."""
    out: Dict[str, Path] = {}
    if "external_files" not in params_df.columns:
        return out

    raw = params_df["external_files"].iloc[0]
    items = None

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for k, v in raw.items():
            key = str(k).strip().upper()
            if key in {"VL", "VR", "LL", "LR"} and v:
                out[key] = _resolve_path(v)
        return out
    elif isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    items = parsed
            except Exception:
                items = [text]
        elif text:
            items = [text]

    if items is None:
        return out

    for item in items:
        s = str(item).strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip().upper()
        val = v.strip()
        if key in {"VL", "VR", "LL", "LR"} and val:
            out[key] = _resolve_path(val)

    return out


def _load_static_irregularity(params_df: pd.DataFrame, cfg: Config) -> Optional[dict]:
    """Load static irregularity channel from external_files and convert amplitude mm -> m."""
    ext_map = _parse_external_file_map(params_df)
    if not ext_map:
        return None

    key = "VR" if "right" in cfg.measured_channel.lower() else "VL"
    path = ext_map.get(key)
    if path is None or (not path.exists()):
        return None

    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None

    x_km = np.asarray(arr[:, 0], dtype=float)
    y_m = np.asarray(arr[:, 1], dtype=float) / 1000.0
    valid = np.isfinite(x_km) & np.isfinite(y_m)
    x_km = x_km[valid]
    y_m = y_m[valid]
    if len(x_km) < 2:
        return None

    order = np.argsort(x_km)
    x_km = x_km[order]
    y_m = y_m[order]
    x_unique, first_idx = np.unique(x_km, return_index=True)
    y_unique = y_m[first_idx]

    return {
        "key": key,
        "path": path,
        "x_km": x_unique,
        "y_m": y_unique,
    }


def _extract_vehicle_geometry(params_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Extract key geometry in meters from argparse params json DataFrame."""
    lc = None
    lt = None
    try:
        if "veh_1" in params_df.columns:
            veh_cfg = params_df["veh_1"].iloc[0]
            if isinstance(veh_cfg, dict):
                if "Lc" in veh_cfg:
                    lc = float(veh_cfg["Lc"])
                if "Lt" in veh_cfg:
                    lt = float(veh_cfg["Lt"])
    except Exception:
        pass

    # 回退1：直接列（若存在）
    try:
        if lc is None and "Lc" in params_df.columns:
            lc = float(params_df["Lc"].iloc[0])
        if lt is None and "Lt" in params_df.columns:
            lt = float(params_df["Lt"].iloc[0])
    except Exception:
        pass

    # 回退2：从 vehicle_params.yaml 读取（按 vehicle_type 匹配）
    if (lc is None or lt is None) and (yaml is not None):
        try:
            vehicle_type = str(params_df["vehicle_type"].iloc[0]).strip() if "vehicle_type" in params_df.columns else ""
            profile_dir = (
                str(params_df["param_profile_dir"].iloc[0]).strip()
                if "param_profile_dir" in params_df.columns
                else "configs/standard"
            )

            candidates = [
                _resolve_path(Path(profile_dir) / "vehicle_params.yaml"),
                _resolve_path(Path("configs/standard/vehicle_params.yaml")),
            ]

            def _find_vehicle_cfg(obj, key_name: str):
                if isinstance(obj, dict):
                    if key_name in obj and isinstance(obj[key_name], dict):
                        return obj[key_name]
                    for v in obj.values():
                        r = _find_vehicle_cfg(v, key_name)
                        if r is not None:
                            return r
                return None

            for p in candidates:
                if not p.exists():
                    continue
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                veh_cfg = _find_vehicle_cfg(data, vehicle_type) if vehicle_type else None
                if isinstance(veh_cfg, dict):
                    if lc is None and "Lc" in veh_cfg:
                        lc = float(veh_cfg["Lc"])
                    if lt is None and "Lt" in veh_cfg:
                        lt = float(veh_cfg["Lt"])
                if lc is not None and lt is not None:
                    break
        except Exception:
            pass

    return {"Lc_m": lc, "Lt_m": lt}


def _infer_static_offsets_m(cfg: Config, params_df: pd.DataFrame) -> Tuple[float, ...]:
    """Infer sampling offsets for static irregularity according to compared simulation channel."""
    if cfg.static_compare_offsets_m is not None and len(cfg.static_compare_offsets_m) > 0:
        return tuple(float(v) for v in cfg.static_compare_offsets_m)

    ch = str(cfg.sim_channel).lower()
    geom = _extract_vehicle_geometry(params_df)
    lc = geom["Lc_m"]
    lt = geom["Lt_m"]

    # 无几何信息时退化为无偏移
    if lc is None or lt is None:
        return (0.0,)

    # 四轴绝对位置（相对车辆中心）
    axle_offsets = (lc + lt, lc - lt, -lc + lt, -lc - lt)

    if ch in {"bogie_pair_axle_z_mean", "wheelset_all_axle_z_mean"}:
        return axle_offsets
    if ch == "bogie_front_axle_z_mean":
        return (lc + lt, lc - lt)
    if ch == "bogie_rear_axle_z_mean":
        return (-lc + lt, -lc - lt)

    # 关键词兜底：如果用户自定义了“bogie + mean”的通道名，优先按前后转向架中心均值处理
    if ("bogie" in ch) and ("mean" in ch):
        if "front" in ch and "rear" in ch:
            return (lc, -lc)
        if "pair" in ch:
            return (lc, -lc)
        if "front" in ch:
            return (lc,)
        if "rear" in ch:
            return (-lc,)

    return (0.0,)


def _sample_static_with_offsets(axis_km: np.ndarray, sx: np.ndarray, sy: np.ndarray, offsets_m: Tuple[float, ...]) -> Optional[np.ndarray]:
    """Sample static irregularity at shifted positions and average them."""
    sampled = []
    for off_m in offsets_m:
        query_km = axis_km + float(off_m) / 1000.0
        if query_km[0] < sx[0] or query_km[-1] > sx[-1]:
            return None
        sampled.append(np.interp(query_km, sx, sy))
    if not sampled:
        return None
    return np.mean(np.vstack(sampled), axis=0)


def align_and_compare(cfg: Config) -> Tuple[dict, pd.DataFrame, dict]:
    data = np.load(cfg.sim_npz_path)
    params_df = pd.read_json(cfg.sim_json_path)
    measured_df = pd.read_csv(cfg.measured_csv_path)

    x = data["X"]
    dt = float(data["dt"])

    sim_channels = build_sim_channels(x, params_df)
    meas_mileage, meas_channels = build_measured_channels(measured_df)
    static_ref = _load_static_irregularity(params_df, cfg) if cfg.include_static_irr else None
    static_offsets_m = _infer_static_offsets_m(cfg, params_df)

    if cfg.sim_channel not in sim_channels:
        raise KeyError(f"Unknown simulation channel: {cfg.sim_channel}")
    if cfg.measured_channel not in meas_channels:
        raise KeyError(f"Unknown measured channel: {cfg.measured_channel}")

    sim_signal = np.asarray(sim_channels[cfg.sim_channel], dtype=float)
    measured_signal_m = np.asarray(meas_channels[cfg.measured_channel], dtype=float) / 1000.0

    start_mileage_km = float(params_df["start_mileage"].iloc[0])
    speed_kmh = float(params_df["vx_set"].iloc[0])
    speed_mps = speed_kmh / 3.6

    sim_t = np.arange(len(sim_signal), dtype=float) * dt
    valid_t = sim_t >= cfg.lead_time_s
    if np.count_nonzero(valid_t) < 10:
        raise ValueError("Too few simulation points remain after lead-time trimming.")

    sim_mileage_all = start_mileage_km + speed_kmh * sim_t / 3600.0
    sim_mileage_cut = sim_mileage_all[valid_t]
    sim_signal_cut = sim_signal[valid_t]

    spacing_km = cfg.target_spacing_m / 1000.0
    sim_mileage_ds = np.arange(sim_mileage_cut[0], sim_mileage_cut[-1] + 1e-12, spacing_km)
    sim_ds = np.interp(sim_mileage_ds, sim_mileage_cut, sim_signal_cut)
    sim_rel_km = sim_mileage_ds - sim_mileage_ds[0]

    dt_cmp = cfg.target_spacing_m / speed_mps
    fs_cmp = 1.0 / dt_cmp

    valid_meas = np.isfinite(meas_mileage) & np.isfinite(measured_signal_m)
    m_km = meas_mileage[valid_meas]
    y_m = measured_signal_m[valid_meas]

    order = np.argsort(m_km)
    m_km = m_km[order]
    y_m = y_m[order]

    m_unique, first_idx = np.unique(m_km, return_index=True)
    y_unique = y_m[first_idx]
    if len(m_unique) < 2:
        raise ValueError("Measured mileage has insufficient valid points.")

    center = float(sim_mileage_ds[0])
    candidate_starts = m_unique[
        (m_unique >= center - cfg.search_window_m / 1000.0)
        & (m_unique <= center + cfg.search_window_m / 1000.0)
    ]
    if len(candidate_starts) == 0:
        raise ValueError("No valid measured start candidates in the search window.")

    if len(candidate_starts) > cfg.max_start_candidates:
        step = int(np.ceil(len(candidate_starts) / cfg.max_start_candidates))
        candidate_starts = candidate_starts[::step]

    sim_raw = sim_ds - np.mean(sim_ds)
    sim_lp = lowpass_filter(sim_raw, fs_cmp, dt_cmp, cfg.lowpass_cutoff_hz)

    best = None
    records = []

    for s0 in candidate_starts:
        measured_axis = s0 + sim_rel_km
        if measured_axis[0] < m_unique[0] or measured_axis[-1] > m_unique[-1]:
            continue

        measured_raw = np.interp(measured_axis, m_unique, y_unique)
        measured_raw = measured_raw - np.mean(measured_raw)
        measured_lp = lowpass_filter(measured_raw, fs_cmp, dt_cmp, cfg.lowpass_cutoff_hz)

        static_raw = None
        static_lp = None
        corr_static_raw = np.nan
        corr_static_lp = np.nan
        nrmse_static_raw = np.nan
        nrmse_static_lp = np.nan

        if static_ref is not None:
            sx = static_ref["x_km"]
            sy = static_ref["y_m"]
            static_raw_interp = _sample_static_with_offsets(measured_axis, sx, sy, static_offsets_m)
            if static_raw_interp is not None:
                static_raw = static_raw_interp
                static_raw = static_raw - np.mean(static_raw)
                static_lp = lowpass_filter(static_raw, fs_cmp, dt_cmp, cfg.lowpass_cutoff_hz)

                corr_static_raw, _, nrmse_static_raw = compute_metrics(sim_raw, static_raw)
                corr_static_lp, _, nrmse_static_lp = compute_metrics(sim_lp, static_lp)

        corr_raw, rmse_raw, nrmse_raw = compute_metrics(sim_raw, measured_raw)
        corr_lp, rmse_lp, nrmse_lp = compute_metrics(sim_lp, measured_lp)
        score = corr_lp - 0.15 * nrmse_lp

        rec = {
            "sim_channel": cfg.sim_channel,
            "measured_channel": cfg.measured_channel,
            "start_km": float(s0),
            "shift_m": float((s0 - center) * 1000.0),
            "corr_raw": corr_raw,
            "rmse_raw_m": rmse_raw,
            "nrmse_raw": nrmse_raw,
            "corr_lp": corr_lp,
            "rmse_lp_m": rmse_lp,
            "nrmse_lp": nrmse_lp,
            "corr_static_raw": float(corr_static_raw),
            "nrmse_static_raw": float(nrmse_static_raw),
            "corr_static_lp": float(corr_static_lp),
            "nrmse_static_lp": float(nrmse_static_lp),
            "score": float(score),
        }
        records.append(rec)

        if (best is None) or (score > best["score"]):
            best = {
                **rec,
                "axis_km": measured_axis,
                "sim_raw": sim_raw,
                "measured_raw": measured_raw,
                "sim_lp": sim_lp,
                "measured_lp": measured_lp,
                "static_raw": static_raw,
                "static_lp": static_lp,
            }

    if best is None:
        raise RuntimeError("No valid alignment candidate produced a usable comparison.")

    scan_df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)

    analysis = {
        "dt_original_s": dt,
        "fs_original_hz": 1.0 / dt,
        "dt_compare_s": dt_cmp,
        "fs_compare_hz": fs_cmp,
        "speed_kmh": speed_kmh,
        "n_points_compare": len(sim_raw),
        "scipy_available": SCIPY_AVAILABLE,
        "static_loaded": static_ref is not None,
        "static_channel": static_ref["key"] if static_ref is not None else None,
        "static_path": str(static_ref["path"]) if static_ref is not None else None,
        "static_offsets_m": tuple(float(v) for v in static_offsets_m),
    }

    return best, scan_df, analysis


def make_figures(best: dict, cfg: Config, analysis: dict) -> Dict[str, plt.Figure]:
    """Create separate comparison figures with legend/text on right-side panel."""
    axis_km = best["axis_km"]
    sim_raw_mm = best["sim_raw"] * 1000.0
    measured_raw_mm = best["measured_raw"] * 1000.0
    sim_lp_mm = best["sim_lp"] * 1000.0
    measured_lp_mm = best["measured_lp"] * 1000.0
    static_raw = best.get("static_raw")
    static_lp = best.get("static_lp")
    static_raw_mm = static_raw * 1000.0 if static_raw is not None else None
    static_lp_mm = static_lp * 1000.0 if static_lp is not None else None

    freq_s, power_s = fft_power(best["sim_raw"], analysis["dt_compare_s"])
    freq_m, power_m = fft_power(best["measured_raw"], analysis["dt_compare_s"])
    amp_s = np.sqrt(power_s)
    amp_m = np.sqrt(power_m)

    # 避免高饱和亮色，统一采用深色/中性色
    color_sim = "#1f4e79"  # deep blue
    color_mea = "#7a1f2b"  # muted dark red
    color_sta = "#4d4d4d"  # dark gray for static irregularity

    def _new_canvas() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        fig = plt.figure(
            figsize=(cfg.figure_width_in * 1.25, cfg.figure_height_in * 0.62),
            dpi=cfg.figure_dpi,
            constrained_layout=True,
        )
        gs = fig.add_gridspec(1, 2, width_ratios=[4.9, 2.1], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax_side = fig.add_subplot(gs[0, 1])
        ax_side.axis("off")
        return fig, ax, ax_side

    def _draw_side_legend_and_text(ax: plt.Axes, ax_side: plt.Axes, text: str = "") -> None:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax_side.legend(
                handles,
                labels,
                loc="upper left",
                frameon=False,
                borderaxespad=0.0,
                handlelength=2.2,
                labelspacing=0.5,
            )
        if text:
            ax_side.text(
                0.0,
                0.50,
                text,
                transform=ax_side.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )

    sim_raw_text = (
        f"Raw corr: {best['corr_raw']:.3f}\n"
        f"Raw NRMSE: {best['nrmse_raw']:.3f}\n"
        f"Shift: {best['shift_m']:.1f} m"
    )
    sim_lp_text = f"LP corr: {best['corr_lp']:.3f}\nLP NRMSE: {best['nrmse_lp']:.3f}\nfc: {cfg.lowpass_cutoff_hz:.1f} Hz"

    static_raw_text = (
        f"Raw corr: {best['corr_static_raw']:.3f}\n"
        f"Raw NRMSE: {best['nrmse_static_raw']:.3f}\n"
        f"Offsets: {analysis.get('static_offsets_m')} m"
    )
    static_lp_text = (
        f"LP corr: {best['corr_static_lp']:.3f}\n"
        f"LP NRMSE: {best['nrmse_static_lp']:.3f}\n"
        f"fc: {cfg.lowpass_cutoff_hz:.1f} Hz"
    )

    # (a) 仿真 vs 实测（Raw）
    fig_raw_sim, ax_raw_sim, side_raw_sim = _new_canvas()
    ax_raw_sim.plot(axis_km, sim_raw_mm, color=color_sim, lw=1.0, label="Simulation (raw)")
    ax_raw_sim.plot(axis_km, measured_raw_mm, color=color_mea, lw=1.0, alpha=0.9, label="Measured (raw)")
    ax_raw_sim.set_xlabel("Mileage (km)")
    ax_raw_sim.set_ylabel("Dynamic irregularity (mm)")
    ax_raw_sim.set_title("Simulation vs measured (raw)", loc="left")
    ax_raw_sim.grid(True)
    ax_raw_sim.tick_params(direction="in", length=3.0, width=0.8)
    _draw_side_legend_and_text(ax_raw_sim, side_raw_sim, sim_raw_text)

    # (b) 仿真 vs 实测（Low-pass）
    fig_lp_sim, ax_lp_sim, side_lp_sim = _new_canvas()
    ax_lp_sim.plot(axis_km, sim_lp_mm, color=color_sim, lw=1.0, label="Simulation (low-pass)")
    ax_lp_sim.plot(axis_km, measured_lp_mm, color=color_mea, lw=1.0, alpha=0.9, label="Measured (low-pass)")
    ax_lp_sim.set_xlabel("Mileage (km)")
    ax_lp_sim.set_ylabel("Dynamic irregularity (mm)")
    ax_lp_sim.set_title("Simulation vs measured (low-pass)", loc="left")
    ax_lp_sim.grid(True)
    ax_lp_sim.tick_params(direction="in", length=3.0, width=0.8)
    _draw_side_legend_and_text(ax_lp_sim, side_lp_sim, sim_lp_text)

    # (c) 静态 vs 实测（Raw）
    fig_raw_static = None
    fig_lp_static = None
    if static_raw_mm is not None:
        fig_raw_static, ax_raw_static, side_raw_static = _new_canvas()
        ax_raw_static.plot(axis_km, static_raw_mm, color=color_sta, lw=1.0, ls="--", alpha=0.95, label="Static irr (raw)")
        ax_raw_static.plot(axis_km, measured_raw_mm, color=color_mea, lw=1.0, alpha=0.9, label="Measured (raw)")
        ax_raw_static.set_xlabel("Mileage (km)")
        ax_raw_static.set_ylabel("Dynamic irregularity (mm)")
        ax_raw_static.set_title("Static vs measured (raw)", loc="left")
        ax_raw_static.grid(True)
        ax_raw_static.tick_params(direction="in", length=3.0, width=0.8)
        _draw_side_legend_and_text(ax_raw_static, side_raw_static, static_raw_text)

    # (d) 静态 vs 实测（Low-pass）
    if static_lp_mm is not None:
        fig_lp_static, ax_lp_static, side_lp_static = _new_canvas()
        ax_lp_static.plot(axis_km, static_lp_mm, color=color_sta, lw=1.0, ls="--", alpha=0.95, label="Static irr (low-pass)")
        ax_lp_static.plot(axis_km, measured_lp_mm, color=color_mea, lw=1.0, alpha=0.9, label="Measured (low-pass)")
        ax_lp_static.set_xlabel("Mileage (km)")
        ax_lp_static.set_ylabel("Dynamic irregularity (mm)")
        ax_lp_static.set_title("Static vs measured (low-pass)", loc="left")
        ax_lp_static.grid(True)
        ax_lp_static.tick_params(direction="in", length=3.0, width=0.8)
        _draw_side_legend_and_text(ax_lp_static, side_lp_static, static_lp_text)

    # (c) FFT
    fig_fft, ax_fft, side_fft = _new_canvas()
    ax_fft.plot(freq_s, amp_s, color=color_sim, lw=1.0, label="Simulation")
    ax_fft.plot(freq_m, amp_m, color=color_mea, lw=1.0, alpha=0.9, label="Measured")
    ax_fft.axvline(cfg.lowpass_cutoff_hz, color="#444444", ls="--", lw=0.8, label="Cutoff")
    ax_fft.set_xlim(0.0, min(100.0, analysis["fs_compare_hz"] / 2.0))
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude (a.u.)")
    ax_fft.set_title("FFT amplitude spectrum", loc="left")
    ax_fft.grid(True)
    ax_fft.tick_params(direction="in", length=3.0, width=0.8)
    _draw_side_legend_and_text(ax_fft, side_fft, f"fc: {cfg.lowpass_cutoff_hz:.1f} Hz")

    # (d) PSD
    fig_psd, ax_psd, side_psd = _new_canvas()
    if SCIPY_AVAILABLE:
        nperseg = min(4096, len(best["sim_raw"]))
        fw_s, psd_s = welch(best["sim_raw"], fs=analysis["fs_compare_hz"], nperseg=nperseg)
        fw_m, psd_m = welch(best["measured_raw"], fs=analysis["fs_compare_hz"], nperseg=nperseg)
        ax_psd.semilogy(fw_s, psd_s + 1e-20, color=color_sim, lw=1.0, label="Simulation")
        ax_psd.semilogy(fw_m, psd_m + 1e-20, color=color_mea, lw=1.0, alpha=0.9, label="Measured")
    else:
        ax_psd.semilogy(freq_s, power_s + 1e-20, color=color_sim, lw=1.0, label="Simulation")
        ax_psd.semilogy(freq_m, power_m + 1e-20, color=color_mea, lw=1.0, alpha=0.9, label="Measured")
    ax_psd.axvline(cfg.lowpass_cutoff_hz, color="#444444", ls="--", lw=0.8, label="Cutoff")
    ax_psd.set_xlim(0.0, min(100.0, analysis["fs_compare_hz"] / 2.0))
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD")
    ax_psd.set_title("Power spectral density", loc="left")
    ax_psd.grid(True)
    ax_psd.tick_params(direction="in", length=3.0, width=0.8)
    _draw_side_legend_and_text(ax_psd, side_psd, f"Scipy PSD: {analysis['scipy_available']}")

    out = {
        "sim_vs_measured_raw": fig_raw_sim,
        "sim_vs_measured_lowpass": fig_lp_sim,
        "fft": fig_fft,
        "psd": fig_psd,
    }
    if fig_raw_static is not None:
        out["static_vs_measured_raw"] = fig_raw_static
    if fig_lp_static is not None:
        out["static_vs_measured_lowpass"] = fig_lp_static
    return out


def main() -> None:
    cfg = Config()
    setup_publication_style()

    best, scan_df, analysis = align_and_compare(cfg)

    print("=" * 72)
    print("Dynamic Irregularity Comparison Summary")
    print("=" * 72)
    print(f"Simulation channel      : {cfg.sim_channel}")
    print(f"Measured channel        : {cfg.measured_channel}")
    print(f"Lead-time trim          : {cfg.lead_time_s:.2f} s")
    print(f"Target spacing          : {cfg.target_spacing_m:.3f} m")
    print(f"Best aligned start      : {best['start_km']:.6f} km")
    print(f"Shift vs simulation     : {best['shift_m']:.2f} m")
    print(f"Raw correlation         : {best['corr_raw']:.4f}")
    print(f"Raw NRMSE               : {best['nrmse_raw']:.4f}")
    print(f"Low-pass correlation    : {best['corr_lp']:.4f}")
    print(f"Low-pass NRMSE          : {best['nrmse_lp']:.4f}")
    if analysis.get("static_loaded", False):
        print(f"Static channel          : {analysis.get('static_channel')}")
        print(f"Static file             : {analysis.get('static_path')}")
        print(f"Static offsets (m)      : {analysis.get('static_offsets_m')}")
        print(f"Static raw correlation  : {best['corr_static_raw']:.4f}")
        print(f"Static raw NRMSE        : {best['nrmse_static_raw']:.4f}")
        print(f"Static LP correlation   : {best['corr_static_lp']:.4f}")
        print(f"Static LP NRMSE         : {best['nrmse_static_lp']:.4f}")
    else:
        print("Static irregularity     : not loaded")
    print(f"Scipy filter/PSD        : {analysis['scipy_available']}")
    print("=" * 72)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    scan_path = cfg.output_dir / f"{cfg.output_stem}_alignment_scan.csv"
    scan_df.to_csv(scan_path, index=False, encoding="utf-8-sig")
    print(f"Alignment scan table saved to: {scan_path}")

    summary_df, band_df = build_improvement_tables(best, cfg, analysis)
    summary_path = cfg.output_dir / f"{cfg.output_stem}_improvement_summary.csv"
    band_path = cfg.output_dir / f"{cfg.output_stem}_band_comparison.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    band_df.to_csv(band_path, index=False, encoding="utf-8-sig")
    print(f"Improvement summary saved to: {summary_path}")
    print(f"Band comparison saved to: {band_path}")

    fig_improve = make_improvement_figure(summary_df, band_df, cfg)

    figures = make_figures(best, cfg, analysis)

    if cfg.export_png:
        for key, fig in figures.items():
            png_path = cfg.output_dir / f"{cfg.output_stem}_{key}.png"
            fig.savefig(png_path, dpi=cfg.export_dpi)
            print(f"Figure saved to: {png_path}")
        png_path = cfg.output_dir / f"{cfg.output_stem}_improvement.png"
        fig_improve.savefig(png_path, dpi=cfg.export_dpi)
        print(f"Figure saved to: {png_path}")

    if cfg.export_pdf:
        for key, fig in figures.items():
            pdf_path = cfg.output_dir / f"{cfg.output_stem}_{key}.pdf"
            fig.savefig(pdf_path)
            print(f"Figure saved to: {pdf_path}")
        pdf_path = cfg.output_dir / f"{cfg.output_stem}_improvement.pdf"
        fig_improve.savefig(pdf_path)
        print(f"Figure saved to: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()