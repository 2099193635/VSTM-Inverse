"""
test_green_function.py
======================
测试格林函数（Green's Function / 脉冲响应函数）
在实现"轨道不平顺激励 → 车辆动力学响应"映射中的可行性。

原理
----
对于线性时不变（LTI）系统：
    y(t) = ∫ g(τ) · u(t-τ) dτ  ←→  Y(ω) = H(ω) · U(ω)

其中:
    u(t)  ── 不平顺激励（输入，轨道高低不平顺）
    y(t)  ── 车辆响应（输出，加速度/位移）
    g(t)  ── 脉冲响应函数（时域格林函数）
    H(ω)  ── 频率响应函数（FRF，频域格林函数）

测试方案
--------
1. 从 VTCM_vertical 数据集载入真实仿真数据
2. 在训练段（前 60%）用最小二乘法辨识经验 FRF
3. 在验证段（后 40%）用格林函数卷积预测响应
4. 与神经网络相比较：计算 RMSE / 相关系数 / 谱误差
5. 对 LTI 假设有效性做定量评估（非线性度分析）

数据结构（VTCM_vertical/validation_full_seq.hdf5）
--------------------------------------------------
  input  : [N, 2, T]  ── 左/右钢轨竖向不平顺 (m)
  output : [N, 21, T] ── 全车 21 DOF 响应 (m 或 m/s²)
  vx_mps : [N]        ── 运行速度 (m/s)
  dt     : [N]        ── 时间步长 (s)
  vehicle_params : [N, 40]
"""

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")            # 无头环境 / 可改为 TkAgg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.linalg import lstsq

# ── 路径 ──────────────────────────────────────────────────────────────────────
_DIR  = Path(__file__).parent
_ROOT = _DIR.parent
DATASET_PATH = _ROOT / "datasets" / "VTCM_vertical" / "validation_full_seq.hdf5"
OUT_DIR = _DIR / "green_function_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 参数 ──────────────────────────────────────────────────────────────────────
SAMPLE_IDX   = 0      # 测试哪个仿真样本
INPUT_CH     = 0      # 激励通道：0=左轨竖向不平顺
OUTPUT_CH    = 0      # 响应通道：0=车体竖向 DOF
TRAIN_RATIO  = 0.60   # 用前 60% 辨识 FRF
FRF_N_AVG    = 8      # Welch 法辨识 FRF 时的平均段数（越多越平滑）
NPERSEG_COEF = 4      # nperseg = T // FRF_N_AVG // NPERSEG_COEF（段长控制）
SMOOTH_WIN   = 5      # FRF 频率轴平滑窗口（奇数，1=不平滑）
HP_CUTOFF    = 0.5    # 预处理高通截止频率 [Hz]，去除低频漂移/列车重力趋势

# DOF 标签（21 DOF 全车模型）
DOF_LABELS = [
    "Body-Z",    "Body-Y",    "Body-Roll",
    "Bogie1-Z",  "Bogie1-Y",  "Bogie1-Roll",
    "Bogie2-Z",  "Bogie2-Y",  "Bogie2-Roll",
    "Axle1-Z",   "Axle1-Y",   "Axle1-Roll",
    "Axle2-Z",   "Axle2-Y",   "Axle2-Roll",
    "Axle3-Z",   "Axle3-Y",   "Axle3-Roll",
    "Axle4-Z",   "Axle4-Y",   "Axle4-Roll",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def load_sample(hdf5_path: Path, idx: int) -> dict:
    """从 HDF5 加载单个仿真样本，返回 numpy 字典。"""
    with h5py.File(hdf5_path, "r") as f:
        sample = {
            "input":          np.array(f["input"][idx],          dtype=np.float64),   # [2, T]
            "output":         np.array(f["output"][idx],         dtype=np.float64),   # [21, T]
            "vx_mps":         float(f["vx_mps"][idx]),
            "dt":             float(f["dt"][idx]),
            "vehicle_params": np.array(f["vehicle_params"][idx], dtype=np.float64),   # [40]
            "line_params":    np.array(f["line_params"][idx],    dtype=np.float64),
        }
    print(f"[数据载入] sample={idx}, vx={sample['vx_mps']:.2f} m/s, "
          f"dt={sample['dt']*1e3:.4f} ms, T={sample['input'].shape[1]}")
    return sample


def empirical_frf_welch(u: np.ndarray, y: np.ndarray, dt: float,
                         n_avg: int = 8, reg: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    用 Welch 互谱法辨识经验 FRF：
        H(f) = Syu(f) / (Suu(f) + reg * max(Suu))

    加入正则化项 reg 防止 Suu 趋近零时 H 数值爆炸。
    返回 (freqs [Hz], H_complex)。
    """
    T   = len(u)
    nperseg = max(512, T // n_avg)
    noverlap = nperseg // 2

    f, Syu = signal.csd(u, y, fs=1.0 / dt, nperseg=nperseg,
                        noverlap=noverlap, window="hann")
    _, Suu = signal.welch(u, fs=1.0 / dt, nperseg=nperseg,
                          noverlap=noverlap, window="hann")

    # 正则化：防止除零/数值爆炸
    reg_val = reg * float(np.max(Suu) + 1e-30)
    H = Syu / (Suu + reg_val)
    return f, H


def coherence(u: np.ndarray, y: np.ndarray, dt: float, n_avg: int = 8) -> tuple:
    """计算 MSC（幅值平方相干函数）。"""
    T   = len(u)
    nperseg = max(512, T // n_avg)
    noverlap = nperseg // 2
    f, coh = signal.coherence(u, y, fs=1.0 / dt, nperseg=nperseg,
                               noverlap=noverlap, window="hann")
    return f, coh


def frf_to_impulse(H: np.ndarray, dt: float) -> np.ndarray:
    """
    将单边 FRF H[f] 转换为时域脉冲响应函数 g(t)。
    通过对称补全为双边谱后做 IFFT。
    """
    N = (len(H) - 1) * 2        # 原始 FFT 长度（假设 T 为偶数）
    H_full = np.zeros(N, dtype=complex)
    H_full[: len(H)]  = H
    H_full[len(H) :]  = np.conj(H[-2:0:-1])   # 共轭对称
    g = np.real(np.fft.ifft(H_full)) / dt      # 量纲修正
    return g


def green_predict(g: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    用脉冲响应函数 g(t) 卷积输入 u(t) 预测响应：
        y_pred = conv(g, u, mode='full')[:T] * dt
    乘以 dt 是将离散卷积还原为黎曼和近似的连续时间积分：
        y(t) = ∫ g(τ) u(t-τ) dτ  ≈  Σ g[k] u[n-k] * dt
    若不乘 dt，预测幅值将偏大 fs 倍（本例为 10000 倍）。
    """
    T   = len(u)
    L_g = len(g)
    # 频域卷积（FFT 卷积，比直接 convolve 快）
    N   = T + L_g - 1
    Y   = np.fft.rfft(g, n=N) * np.fft.rfft(u, n=N)
    y_pred = np.real(np.fft.irfft(Y, n=N))[:T] * dt   # ← dt 积分步长修正
    return y_pred


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """计算预测质量指标（含幅值归一化 R²）。"""
    mse  = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    corr = float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 1 else float("nan")
    var  = float(np.var(true))
    r2   = 1.0 - mse / var if var > 1e-20 else float("nan")
    rms_true = float(np.sqrt(np.mean(true ** 2)))
    nrmse = rmse / rms_true if rms_true > 1e-20 else float("nan")

    # 幅值归一化后 R²（只比较形状/相位，消除幅值偏差）
    # 将 pred 缩放到与 true 相同 RMS 后再计算 R²
    rms_pred = float(np.sqrt(np.mean(pred ** 2)))
    if rms_pred > 1e-20 and rms_true > 1e-20:
        pred_scaled = pred * (rms_true / rms_pred)
        mse_scaled  = float(np.mean((pred_scaled - true) ** 2))
        r2_norm     = 1.0 - mse_scaled / var if var > 1e-20 else float("nan")
    else:
        r2_norm = float("nan")

    return dict(rmse=rmse, corr=corr, r2=r2, nrmse=nrmse, r2_norm=r2_norm)


def smooth(arr: np.ndarray, win: int) -> np.ndarray:
    """一维均值平滑（奇数窗）。"""
    if win <= 1:
        return arr
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode="same")


def preprocess(sig: np.ndarray, dt: float,
               hp_cutoff: float = 0.5) -> np.ndarray:
    """
    预处理：去均值 + 高通滤波，消除低频漂移/趋势。
    hp_cutoff: 高通截止频率 [Hz]，默认 0.5 Hz。
    """
    sig = sig - np.mean(sig)   # 去直流
    fs  = 1.0 / dt
    nyq = fs / 2.0
    if hp_cutoff > 0 and hp_cutoff < nyq:
        b, a = signal.butter(4, hp_cutoff / nyq, btype="high")
        sig  = signal.filtfilt(b, a, sig)
    return sig


def to_acceleration(disp: np.ndarray, dt: float, lp_cutoff: float = 200.0) -> np.ndarray:
    """
    位移 → 加速度：时域二次差分后低通滤波。
    去除高频量化噪声放大效应。
    """
    vel  = np.gradient(disp, dt)
    acc  = np.gradient(vel,  dt)
    # 低通防混叠
    fs  = 1.0 / dt
    nyq = fs / 2.0
    if lp_cutoff > 0 and lp_cutoff < nyq:
        b, a = signal.butter(4, lp_cutoff / nyq, btype="low")
        acc  = signal.filtfilt(b, a, acc)
    return acc


def extract_irregularity(rail_abs: np.ndarray, dt: float,
                         smooth_window_s: float = 1.0) -> np.ndarray:
    """
    从绝对钢轨高程中提取不平顺偏差：
        irregularity = rail_abs - rolling_mean(rail_abs, smooth_window)
    rolling_mean 滤除坡度/曲线设计线形，保留毫米级动态不平顺。
    """
    win = max(3, int(smooth_window_s / dt))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    trend  = np.convolve(rail_abs, kernel, mode="same")
    # 边界修正（边缘用原信号填充，避免卷积边界畸变）
    half = win // 2
    trend[:half]  = trend[half]
    trend[-half:] = trend[-half - 1]
    return rail_abs - trend


# ═══════════════════════════════════════════════════════════════════════════════
# 主测试逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  格林函数可行性测试：轨道不平顺 → 车辆动力学响应")
    print("=" * 60)

    # ── 1. 载入数据 ────────────────────────────────────────────────────────────
    sample = load_sample(DATASET_PATH, SAMPLE_IDX)
    u_L_raw   = sample["input"][0]           # 左轨绝对高程 [T]
    u_R_raw   = sample["input"][1]           # 右轨绝对高程 [T]
    u_all_raw = sample["input"][INPUT_CH]    # 选定单侧（保留原逻辑）
    y_all_raw = sample["output"][OUTPUT_CH]  # 绝对车体位移 [T]
    dt    = sample["dt"]
    vx    = sample["vx_mps"]
    T     = len(u_all_raw)
    fs    = 1.0 / dt
    t_axis = np.arange(T) * dt

    # ── 模态解耦激励 ──────────────────────────────────────────────────────────
    # 垂向沉浮 (Z) 由对称激励驱动：u_sym = (u_L + u_R) / 2
    # 侧滚 (Roll) 由反对称激励驱动：u_asym = u_L - u_R
    u_sym_raw  = (u_L_raw + u_R_raw) / 2.0
    u_asym_raw = u_L_raw - u_R_raw

    print(f"\n[信号概况（原始）]")
    print(f"  总样本数  T = {T}")
    print(f"  采样率    fs = {fs:.1f} Hz")
    print(f"  时长      {T * dt:.2f} s")
    print(f"  速度      {vx:.2f} m/s = {vx*3.6:.1f} km/h")
    print(f"  左轨高程 RMS = {np.sqrt(np.mean(u_L_raw**2)):.4f} m")
    print(f"  右轨高程 RMS = {np.sqrt(np.mean(u_R_raw**2)):.4f} m")
    print(f"  车体绝对位移 RMS = {np.sqrt(np.mean(y_all_raw**2)):.4f} m  <- 含坡度追随")
    print(f"  响应通道  OUTPUT_CH={OUTPUT_CH} -> {DOF_LABELS[OUTPUT_CH]}")

    # ── 2. 信号工程：带通滤波 + 模态解耦 ────────────────────────────────────
    print("\n[信号工程] 带通滤波 (0.5-50 Hz) + 模态解耦 (对称/反对称激励)...")

    # 车辆-轨道振动主频段：0.5~50 Hz
    BP_LOW  = 0.5    # Hz
    BP_HIGH = 50.0   # Hz
    nyq = fs / 2.0

    # 用 SOS 形式避免在极端频率比（0.5/5000=1e-4）下 ba 系数数值溢出
    sos_bp = signal.butter(4, [BP_LOW / nyq, min(BP_HIGH / nyq, 0.99)],
                           btype="band", output="sos")

    # ── Fix: 先切分，再独立带通滤波（防止 sosfiltfilt 反向滤波泄露验证段信息）─
    n_train = int(T * TRAIN_RATIO)

    def _bp(x): return signal.sosfiltfilt(sos_bp, x)

    # 训练段：独立带通（零相位，双向，仅作用于训练数据）
    u_L_tr  = _bp(u_L_raw[:n_train])
    u_R_tr  = _bp(u_R_raw[:n_train])
    y_tr_raw = _bp(y_all_raw[:n_train])
    u_sym_tr  = (u_L_tr + u_R_tr) / 2.0
    u_asym_tr = u_L_tr - u_R_tr

    # 验证段：独立带通（零相位，无法从训练段获取任何信息）
    u_L_val  = _bp(u_L_raw[n_train:])
    u_R_val  = _bp(u_R_raw[n_train:])
    y_val_raw = _bp(y_all_raw[n_train:])
    u_sym_val  = (u_L_val + u_R_val) / 2.0
    u_asym_val = u_L_val - u_R_val

    # 根据 DOF 类型选择激励通道
    # Z/Y DOF（索引 0,1,3,4,6,7,9,10,12,13,15,16,18,19）使用对称激励
    # Roll DOF（索引 2,5,8,11,14,17,20）使用反对称激励
    _is_roll = lambda ch: (ch % 3 == 2)
    u_train = u_asym_tr  if _is_roll(OUTPUT_CH) else u_sym_tr
    u_val   = u_asym_val if _is_roll(OUTPUT_CH) else u_sym_val

    # 响应：带通后相对位移（车体 - 使用左轨基准）
    u_ref_tr  = u_L_tr  if not _is_roll(OUTPUT_CH) else np.zeros_like(u_L_tr)
    u_ref_val = u_L_val if not _is_roll(OUTPUT_CH) else np.zeros_like(u_L_val)
    y_train = y_tr_raw  - u_ref_tr
    y_val   = y_val_raw - u_ref_val

    # 供绘图用的全序列版本（仅对称激励，用于 Figure 1）
    u_sym_full  = _bp(u_sym_raw)
    y_full_bp   = _bp(y_all_raw)
    u_all = u_sym_full
    y_all = y_full_bp - _bp(u_L_raw)
    FMAX_ANALYSIS = BP_HIGH

    print(f"  训练段带通 u_train RMS = {np.sqrt(np.mean(u_train**2))*1e3:.3f} mm")
    print(f"  训练段带通 y_train RMS = {np.sqrt(np.mean(y_train**2))*1e3:.3f} mm")
    print(f"  验证段带通 u_val   RMS = {np.sqrt(np.mean(u_val**2))*1e3:.3f} mm")
    print(f"\n[数据划分]  train={n_train}  val={T - n_train}")

    # ── 3. 辨识经验 FRF ───────────────────────────────────────────────────────
    print("\n[FRF 辨识]  方法=Welch 互谱法")
    f_Hz, H_emp = empirical_frf_welch(u_train, y_train, dt, n_avg=FRF_N_AVG)
    f_coh, coh  = coherence(u_train, y_train, dt, n_avg=FRF_N_AVG)

    mean_coh = float(np.mean(coh))
    high_coh_ratio = float(np.mean(coh > 0.8))
    print(f"  平均 MSC（相干函数）= {mean_coh:.3f}")
    print(f"  MSC > 0.8 的频段占比 = {high_coh_ratio * 100:.1f}%")

    # 判断线性度
    if mean_coh > 0.75:
        print("  → 系统线性度良好，格林函数映射可行性高")
    elif mean_coh > 0.5:
        print("  → 系统存在一定非线性，格林函数为近似描述")
    else:
        print("  → 相干度低，系统非线性/噪声严重，格林函数映射受限")

    # ── 4. FRF → 脉冲响应 g(t) ────────────────────────────────────────────────
    g = frf_to_impulse(H_emp, dt)
    print(f"\n[脉冲响应]  g(t) 长度 = {len(g)} pts = {len(g) * dt:.3f} s")

    # 截断到合理长度（系统衰减后即可，防止尾部噪声）
    # 估计截断点：脉冲响应峰值的 0.1% 处
    g_env = np.abs(signal.hilbert(g))
    peak_val = g_env.max()
    cutoff_idx = np.argmax(g_env < peak_val * 0.001)
    if cutoff_idx == 0:
        cutoff_idx = len(g) // 4   # 若未找到，取 1/4 长度
    g_trunc = g[:max(cutoff_idx, 1024)]   # 至少保留 1024 点
    print(f"  截断脉冲响应：{len(g_trunc)} pts = {len(g_trunc) * dt:.3f} s")

    # ── 5. 验证段格林函数预测 ──────────────────────────────────────────────────
    print("\n[格林函数预测]  在验证段（未见数据）进行")
    y_pred_val = green_predict(g_trunc, u_val, dt)
    # Fix 3: 跳过开头的瞬态建立阶段（约等于脉冲响应长度）
    # 验证段是从连续时序中截取的，t=0 处系统并非静止，
    # 卷积隐式假设静止初始条件，因此开头 len(g_trunc) 点存在瞬态误差
    skip = len(g_trunc)
    m = metrics(y_pred_val[skip:], y_val[skip:])
    print(f"  （跳过开头 {skip} 点 = {skip*dt:.3f} s 的瞬态误差后计算）")
    print(f"  RMSE  = {m['rmse']:.6f}")
    print(f"  nRMSE = {m['nrmse']:.4f}  ({m['nrmse']*100:.1f}% of RMS)")
    print(f"  R²    = {m['r2']:.4f}")
    print(f"  R²_norm = {m['r2_norm']:.4f}")
    print(f"  相关系数 = {m['corr']:.4f}")

    # ── 6. 全序列重构（训练段自洽性检验）─────────────────────────────────────
    print("\n[自洽性检验]  在训练段重构")
    y_pred_train = green_predict(g_trunc, u_train, dt)
    m_train = metrics(y_pred_train[skip:], y_train[skip:])
    print(f"  RMSE  = {m_train['rmse']:.6f}  R² = {m_train['r2']:.4f}  corr = {m_train['corr']:.4f}")

    # ── 7. 全通道 FRF 辨识（批量分析所有 21 DOF）─────────────────────────────
    print("\n[全通道分析]  对全部 21 DOF 输出进行 FRF 辨识")
    n_out = sample["output"].shape[0]
    all_r2, all_corr, all_coh, all_r2_norm = [], [], [], []
    for ch in range(n_out):
        y_ch_raw = sample["output"][ch]
        # 每个 DOF：带通后取相对位移
        # 按 DOF 类型选择解耦激励（模态分解）
        u_tr_ch  = u_asym_tr  if _is_roll(ch) else u_sym_tr
        u_val_ch = u_asym_val if _is_roll(ch) else u_sym_val
        u_ref_tr_ch  = np.zeros(n_train)         if _is_roll(ch) else u_L_tr
        u_ref_val_ch = np.zeros(T - n_train)     if _is_roll(ch) else u_L_val
        y_ch_tr_bp  = _bp(y_ch_raw[:n_train])
        y_ch_val_bp = _bp(y_ch_raw[n_train:])
        y_ch_train  = y_ch_tr_bp  - u_ref_tr_ch
        y_ch_val    = y_ch_val_bp - u_ref_val_ch
        try:
            f_ch, H_ch = empirical_frf_welch(u_tr_ch, y_ch_train, dt, n_avg=FRF_N_AVG)
            _, coh_ch  = coherence(u_tr_ch, y_ch_train, dt, n_avg=FRF_N_AVG)
            g_ch       = frf_to_impulse(H_ch, dt)
            g_ch_env   = np.abs(signal.hilbert(g_ch))
            peak_ch    = g_ch_env.max()
            cut_ch     = np.argmax(g_ch_env < peak_ch * 0.001)
            if cut_ch == 0:
                cut_ch = len(g_ch) // 4
            g_ch_tr    = g_ch[:max(cut_ch, 512)]
            skip_ch    = len(g_ch_tr)
            y_pred_ch  = green_predict(g_ch_tr, u_val_ch, dt)
            m_ch       = metrics(y_pred_ch[skip_ch:], y_ch_val[skip_ch:])
            all_r2.append(m_ch["r2"])
            all_corr.append(m_ch["corr"])
            all_coh.append(float(np.mean(coh_ch)))
            all_r2_norm.append(m_ch["r2_norm"])
        except Exception as e:
            all_r2.append(float("nan"))
            all_corr.append(float("nan"))
            all_coh.append(float("nan"))
            all_r2_norm.append(float("nan"))

    print(f"\n  {'DOF':<14} {'MSC':>7} {'R²(norm)':>10} {'corr':>8}")
    print(f"  {'-'*44}")
    for ch in range(n_out):
        label = DOF_LABELS[ch] if ch < len(DOF_LABELS) else f"DOF{ch}"
        print(f"  {label:<14} {all_coh[ch]:>7.3f} {all_r2_norm[ch]:>10.4f} {all_corr[ch]:>8.4f}")

    valid_r2n = [v for v in all_r2_norm if not np.isnan(v)]
    print(f"\n  幅值归一化后平均 R² = {np.mean(valid_r2n):.4f}  (中位数 {np.median(valid_r2n):.4f})")
    print(f"  R²_norm > 0.8 的通道：{sum(v > 0.8 for v in valid_r2n)} / {n_out}")
    print(f"  R²_norm > 0.5 的通道：{sum(v > 0.5 for v in valid_r2n)} / {n_out}")

    # ── 8. 绘图 ────────────────────────────────────────────────────────────────
    _plot_results(
        u_all, y_all, y_train, y_val, y_pred_train, y_pred_val,
        n_train, dt, f_Hz, H_emp, coh,
        g_trunc, m, all_r2_norm, all_coh, all_corr,
        sample, vx,
    )
    print(f"\n[可视化]  图片保存在 {OUT_DIR}")

    # ── 9. 结论 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  可行性结论")
    print("=" * 60)
    _conclusion(m, all_r2_norm, all_coh, n_out)


# ═══════════════════════════════════════════════════════════════════════════════
# 绘图函数
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_results(u_all, y_all, y_train, y_val, y_pred_train, y_pred_val,
                  n_train, dt, f_Hz, H_emp, coh,
                  g_trunc, m_val, all_r2, all_coh, all_corr,
                  sample, vx):
    """Generate five publication-quality figures and save to disk."""

    # ── Global style ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelweight":   "bold",
        "axes.linewidth":     1.2,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        "legend.fontsize":    9,
        "legend.frameon":     True,
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.7",
        "lines.linewidth":    1.5,
        "figure.dpi":         180,
        "savefig.dpi":        300,
        "axes.prop_cycle":    plt.cycler(color=[
            "#1B4F72", "#C0392B", "#1E8449", "#7D3C98",
            "#D35400", "#1A5276", "#922B21", "#196F3D",
        ]),
    })

    # Colour palette
    C_BLUE   = "#1B4F72"
    C_RED    = "#C0392B"
    C_GREEN  = "#1E8449"
    C_ORANGE = "#D35400"
    C_GRAY   = "#7F8C8D"
    C_TRAIN_BG = "#EAF2FB"
    C_VAL_BG   = "#FDFEFE"

    T      = len(u_all)
    t_all  = np.arange(T) * dt
    t_train = t_all[:n_train]
    t_val   = t_all[n_train:]
    dof_lbl = DOF_LABELS[OUTPUT_CH]

    # ── Figure 1: Time-domain comparison ─────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             constrained_layout=True, sharex=False)

    # Panel A – excitation
    ax = axes[0]
    ax.plot(t_all, u_all, color=C_BLUE, lw=0.8, alpha=0.9,
            label="Rail irregularity $u(t)$")
    ax.axvline(t_train[-1], color=C_GRAY, ls="--", lw=1.2,
               label="Train / Val boundary")
    ax.axvspan(0, t_train[-1], alpha=0.04, color=C_BLUE)
    ax.axvspan(t_train[-1], t_all[-1], alpha=0.04, color=C_RED)
    ax.set_ylabel("Irregularity (m)")
    ax.set_title("(a)  Input Excitation — Rail Irregularity")
    ax.legend(loc="upper right")
    ax.set_xlim([0, t_all[-1]])

    # Panel B – training reconstruction
    ax = axes[1]
    ax.plot(t_train, y_train, color=C_GREEN, lw=0.9, alpha=0.85,
            label=f"Ground truth — {dof_lbl}")
    ax.plot(t_train, y_pred_train, color=C_ORANGE, lw=0.9, ls="--",
            alpha=0.92, label="Green's function prediction (train)")
    ax.axvline(t_train[-1], color=C_GRAY, ls="--", lw=1.2)
    ax.axvspan(0, t_train[-1], alpha=0.04, color=C_BLUE)
    ax.set_ylabel("Response (m)")
    ax.set_title("(b)  Training Segment — Self-Consistency Check")
    ax.legend(loc="upper right")
    ax.set_xlim([0, t_all[-1]])

    # Panel C – validation prediction
    ax = axes[2]
    ax.plot(t_val, y_val, color=C_GREEN, lw=1.0, alpha=0.9,
            label="Ground truth (validation)")
    ax.plot(t_val, y_pred_val, color=C_RED, lw=1.0, ls="--", alpha=0.92,
            label=(f"Green's function  "
                   f"$r^2_{{\\rm norm}}$={m_val['r2_norm']:.3f}  "
                   f"$\\rho$={m_val['corr']:.3f}"))
    ax.axvspan(t_train[-1], t_all[-1], alpha=0.04, color=C_RED)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response (m)")
    ax.set_title("(c)  Validation Segment — Unseen Data Prediction")
    ax.legend(loc="upper right")
    ax.set_xlim([0, t_all[-1]])

    fig.suptitle(
        f"Green's Function Mapping: Rail Irregularity $\\rightarrow$ {dof_lbl}"
        f"   ($v_x$={vx:.1f} m/s = {vx*3.6:.0f} km/h)",
        fontsize=13, fontweight="bold"
    )
    fig.savefig(OUT_DIR / "01_time_domain.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: FRF magnitude + phase + coherence ───────────────────────────
    f_max_plot = min(50.0, f_Hz[-1])
    mask = f_Hz <= f_max_plot

    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                             constrained_layout=True, sharex=True)

    # Magnitude
    ax = axes[0]
    H_amp_dB = 20 * np.log10(np.abs(H_emp) + 1e-12)
    ax.plot(f_Hz[mask], smooth(H_amp_dB[mask], SMOOTH_WIN),
            color=C_BLUE, lw=1.6)
    ax.fill_between(f_Hz[mask], smooth(H_amp_dB[mask], SMOOTH_WIN),
                    H_amp_dB[mask].min() - 5,
                    alpha=0.12, color=C_BLUE)
    ax.set_ylabel("|H(f)| (dB)")
    ax.set_title("(a)  Frequency Response Function — Magnitude")
    ax.grid(True, which="both", ls=":", lw=0.5, color="0.75")

    # Phase
    ax = axes[1]
    H_phase = np.degrees(np.angle(H_emp))
    ax.plot(f_Hz[mask], smooth(H_phase[mask], SMOOTH_WIN),
            color=C_RED, lw=1.6)
    ax.axhline( 180, color=C_GRAY, ls=":", lw=0.8)
    ax.axhline(-180, color=C_GRAY, ls=":", lw=0.8)
    ax.axhline(0,    color=C_GRAY, ls="-", lw=0.6)
    ax.set_ylabel("Phase (deg)")
    ax.set_ylim([-200, 200])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_title("(b)  Frequency Response Function — Phase")
    ax.grid(True, which="both", ls=":", lw=0.5, color="0.75")

    # Coherence
    ax = axes[2]
    coh_plot = coh[mask]
    f_plot   = f_Hz[mask]
    ax.fill_between(f_plot, coh_plot, alpha=0.20, color=C_GREEN)
    ax.plot(f_plot, coh_plot, color=C_GREEN, lw=1.6,
            label=r"MSC $\gamma^2(f)$")
    ax.axhline(0.8, color=C_RED, ls="--", lw=1.2, label="Threshold = 0.8")
    ax.axhline(0.5, color=C_ORANGE, ls=":", lw=1.0, label="Threshold = 0.5")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"MSC $\gamma^2$")
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, f_max_plot])
    ax.set_title("(c)  Magnitude-Squared Coherence (LTI linearity indicator)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls=":", lw=0.5, color="0.75")

    fig.suptitle(
        f"Empirical FRF Identification — {dof_lbl}  (Welch method, training segment)",
        fontsize=13, fontweight="bold"
    )
    fig.savefig(OUT_DIR / "02_frf_identified.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Impulse response function g(t) ──────────────────────────────
    t_g = np.arange(len(g_trunc)) * dt
    env = np.abs(signal.hilbert(g_trunc))

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.fill_between(t_g,  env, -env, alpha=0.12, color=C_BLUE, label="Envelope")
    ax.plot(t_g, g_trunc, color=C_BLUE, lw=1.3, label="$g(t)$", zorder=3)
    ax.plot(t_g,  env, color=C_RED, ls="--", lw=1.0, alpha=0.85)
    ax.plot(t_g, -env, color=C_RED, ls="--", lw=1.0, alpha=0.85,
            label="Hilbert envelope")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Impulse Response Function (Green's Function) — {dof_lbl}")
    ax.legend(loc="upper right")
    ax.grid(True, ls=":", lw=0.5, color="0.75")
    fig.savefig(OUT_DIR / "03_impulse_response.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4: PSD comparison (validation segment) ────────────────────────
    T_val     = len(y_val)
    nperseg_p = min(2048, T_val // 4)
    f_p, psd_true = signal.welch(y_val,      fs=1.0/dt, nperseg=nperseg_p)
    _,   psd_pred = signal.welch(y_pred_val, fs=1.0/dt, nperseg=nperseg_p)
    mask_p = f_p <= min(50.0, f_p[-1])

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.semilogy(f_p[mask_p], psd_true[mask_p],
                color=C_GREEN, lw=1.8, alpha=0.90, label="Ground truth PSD")
    ax.semilogy(f_p[mask_p], psd_pred[mask_p],
                color=C_RED, lw=1.6, ls="--", alpha=0.95,
                label="Green's function prediction PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (m² / Hz)")
    ax.set_title(f"Power Spectral Density Comparison — {dof_lbl}  (Validation segment)")
    ax.set_xlim([0, min(50.0, f_p[-1])])
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls=":", lw=0.5, color="0.75")
    fig.savefig(OUT_DIR / "04_psd_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 5: All-channel bar chart summary ───────────────────────────────
    n_ch    = len(all_r2)
    eng_labels = [
        "Body-Z", "Body-Y", "Body-Roll",
        "Bogie1-Z", "Bogie1-Y", "Bogie1-Roll",
        "Bogie2-Z", "Bogie2-Y", "Bogie2-Roll",
        "Axle1-Z", "Axle1-Y", "Axle1-Roll",
        "Axle2-Z", "Axle2-Y", "Axle2-Roll",
        "Axle3-Z", "Axle3-Y", "Axle3-Roll",
        "Axle4-Z", "Axle4-Y", "Axle4-Roll",
    ]
    labels = [eng_labels[i] if i < len(eng_labels) else f"DOF{i}"
              for i in range(n_ch)]
    x = np.arange(n_ch)

    def _bar_color(v, hi=0.8, lo=0.5):
        if np.isnan(v): return C_GRAY
        if v >= hi: return C_GREEN
        if v >= lo: return C_ORANGE
        return C_RED

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9),
                                   constrained_layout=True)

    # R² (normalised)
    c_r2 = [_bar_color(v) for v in all_r2]
    bars = ax1.bar(x, all_r2, color=c_r2, width=0.65,
                   edgecolor="white", linewidth=0.6)
    ax1.axhline(0.8, color="k",    ls="--", lw=1.2, label="$R^2_{\\rm norm}=0.8$")
    ax1.axhline(0.5, color=C_GRAY, ls=":",  lw=1.0, label="$R^2_{\\rm norm}=0.5$")
    ax1.axhline(0.0, color="k",    ls="-",  lw=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax1.set_ylabel("Normalised $R^2$")
    ax1.set_title(
        "(a)  Green's Function Prediction Quality — Normalised $R^2$ (Validation)"
    )
    y_lo = min(-0.15, min(v for v in all_r2 if not np.isnan(v)) - 0.05)
    ax1.set_ylim([y_lo, 1.10])
    ax1.legend(loc="upper right")
    ax1.grid(True, axis="y", ls=":", lw=0.5, color="0.75")
    # annotate bars
    for bar, val in zip(bars, all_r2):
        if not np.isnan(val):
            ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.06
            ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
                     f"{val:.2f}", ha="center", va="bottom",
                     fontsize=7, color="0.25")

    # MSC
    c_msc = [_bar_color(v) for v in all_coh]
    bars2 = ax2.bar(x, all_coh, color=c_msc, width=0.65,
                    edgecolor="white", linewidth=0.6)
    ax2.axhline(0.8, color="k",    ls="--", lw=1.2, label="MSC = 0.8")
    ax2.axhline(0.5, color=C_GRAY, ls=":",  lw=1.0, label="MSC = 0.5")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax2.set_ylabel("Mean MSC $\\gamma^2$")
    ax2.set_title("(b)  FRF Identification Quality — Mean Coherence (Training segment)")
    ax2.set_ylim([0, 1.12])
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", ls=":", lw=0.5, color="0.75")
    for bar, val in zip(bars2, all_coh):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom",
                     fontsize=7, color="0.25")

    # Colour legend patch
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=C_GREEN,  label="$\\geq 0.8$ (Good)"),
        Patch(color=C_ORANGE, label="$0.5$–$0.8$ (Fair)"),
        Patch(color=C_RED,    label="$< 0.5$ (Poor)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Green's Function Feasibility Assessment — All 21 DOF Summary",
        fontsize=13, fontweight="bold"
    )
    fig.savefig(OUT_DIR / "05_all_channels_summary.png", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 结论打印
# ═══════════════════════════════════════════════════════════════════════════════

def _conclusion(m_val: dict, all_r2: list, all_coh: list, n_out: int):
    valid_r2  = [v for v in all_r2  if not np.isnan(v)]
    valid_coh = [v for v in all_coh if not np.isnan(v)]
    mean_r2   = float(np.mean(valid_r2)) if valid_r2 else float("nan")
    mean_coh  = float(np.mean(valid_coh)) if valid_coh else float("nan")

    print(f"\n  [主输出通道 {DOF_LABELS[OUTPUT_CH]}]")
    print(f"    验证集 R²_norm = {m_val['r2_norm']:.4f}  (幅值归一化后)")
    print(f"    验证集 corr    = {m_val['corr']:.4f}")
    print(f"    说明：相关系数高 = 格林函数能捕捉时序/相位关系；"
          f"R² 负 = 幅值估计存在偏差")

    print(f"\n  [全通道汇总（{n_out} DOF）]")
    print(f"    平均 MSC         = {mean_coh:.3f}  (线性度指标，越接近1越好)")
    print(f"    平均 R²_norm     = {mean_r2:.3f}  (消除幅值偏差后的预测质量)")
    print(f"    R²_norm > 0.8   : {sum(v > 0.8 for v in valid_r2)} / {n_out}")
    print(f"    R²_norm > 0.5   : {sum(v > 0.5 for v in valid_r2)} / {n_out}")

    print("\n  [可行性判断]")
    if mean_r2 > 0.75 and mean_coh > 0.75:
        verdict = "PASS: 高可行性"
        detail  = ("系统线性度高，格林函数可作为轻量级物理代理，"
                   "适合作为神经网络的物理先验或预训练初始化。")
    elif mean_r2 > 0.50 or mean_coh > 0.55:
        verdict = "PARTIAL: 中等可行性"
        detail  = ("格林函数可捕捉主要频率成分和时序模式（相关系数较好），"
                   "但幅值受 LTI 近似和多输入混叠影响而不稳定。\n"
                   "    建议①：格林函数 FRF 用于神经网络的频谱约束损失（FRFForwardConsistencyLoss）。\n"
                   "    建议②：单纯时域卷积预测不可用，但频域 FRF 特征可用。")
    else:
        verdict = "FAIL: 可行性有限"
        detail  = ("非线性或噪声主导，纯格林函数不足以描述动力学映射。"
                   "建议：放弃独立格林函数，仅保留 FRF 约束作为神经网络辅助损失项。")

    print(f"    [{verdict}]")
    print(f"    {detail}")

    print("\n  [对神经网络的启示]")
    print(f"    1. MSC 高频段（本例：轮对 DOF，MSC≈0.8-0.9）→ FRFForwardConsistencyLoss 有效频率范围")
    print(f"    2. MSC 低频段 → 非线性主导，神经网络需从数据学习（不要硬约束）")
    print(f"    3. 本数据仅用了 1 条轨道激励（共 2 条），完整分析应使用双输入 FRF 矩阵")
    print(f"    4. 辨识的 FRF H(f) 可初始化 FRFForwardConsistencyLoss 的 _ModalFRF 权重矩阵")


if __name__ == "__main__":
    main()
