"""
VTCM PINO 数据集加载器

模型输入输出定义：
    - 输入 u(t): [in_channels, time_steps] = [2, T] (左右轨垂向不平顺激励分离)
  - 输出 y(t): [out_channels * 3, time_steps] = [105, T]  其中 105 = 35 DOF × 3 (位移/速度/加速度)
  
数据来源：
  - 从 results/<project>/<run>*/files/simulation_result.npz 加载历史仿真数据
  - 需要已运行 generate_main.py 生成足够的训练片段
"""

import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.parameters import (
    Fastener_KV,
    RealTrackAlignment,
    RailParams,
    Subrail_Params,
    VehicleParams,
)


_TRACK_ALIGNMENT_CACHE: Dict[Tuple[str, str, str, str], RealTrackAlignment] = {}


def _dataclass_to_dict(obj: Any, include_fields: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
    """将参数 dataclass 转为普通字典，便于随样本返回。"""
    if not is_dataclass(obj):
        return {}

    names = include_fields or tuple(f.name for f in fields(obj) if not f.name.startswith("_"))
    out: Dict[str, Any] = {}
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if isinstance(value, np.generic):
            out[name] = value.item()
        elif isinstance(value, np.ndarray):
            out[name] = value.tolist()
        else:
            out[name] = value
    return out


def _load_run_metadata(npz_path: Path) -> Dict[str, Any]:
    """读取每个仿真结果旁边的 argparse_params.json。"""
    meta_path = npz_path.parent / "argparse_params.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"  [警告] 读取 {meta_path} 失败: {exc}")
        return {}


def _get_track_alignment(meta: Dict[str, Any]) -> Optional[RealTrackAlignment]:
    """按线路文件组合缓存线型对象，避免重复读取 CSV/缓存文件。"""
    curve_file_dir = str(meta.get("curve_file_dir", "configs/curve_parameters.csv"))
    gradient_file_dir = str(meta.get("gradient_file_dir", "configs/gradient_parameters.csv"))
    cache_file_dir = str(meta.get("cache_file_dir", "configs/track_cache.npz"))
    force_rebuild = str(meta.get("force_rebuild", "Off"))
    cache_key = (curve_file_dir, gradient_file_dir, cache_file_dir, force_rebuild)

    if cache_key not in _TRACK_ALIGNMENT_CACHE:
        _TRACK_ALIGNMENT_CACHE[cache_key] = RealTrackAlignment(
            curve_file_dir=curve_file_dir,
            gradient_file_dir=gradient_file_dir,
            cache_file_dir=cache_file_dir,
            force_rebuild=force_rebuild,
        )
    return _TRACK_ALIGNMENT_CACHE[cache_key]


def _build_physics_context(npz_path: Path) -> Dict[str, Any]:
    """从 argparse_params.json 恢复耦合动力学所需的物理上下文。"""
    meta = _load_run_metadata(npz_path)
    if not meta:
        return {
            "run_metadata": {},
            "vehicle": {},
            "rail": {},
            "fastener": {},
            "subrail": {},
            "line": {},
        }

    param_profile_dir = str(meta.get("param_profile_dir", "configs/standard"))
    vehicle_type = str(meta.get("vehicle_type", "高速客车"))
    rail_type = str(meta.get("rail_type", "CHN60"))
    fastener_type = str(meta.get("fastener_type", "Standard_KV"))
    start_mileage_m = float(meta.get("start_mileage", 0.0)) * 1000.0

    try:
        vehicle = VehicleParams(vehicle_type=vehicle_type, yaml_dir=param_profile_dir)
        vehicle_ctx = _dataclass_to_dict(vehicle)
    except Exception as exc:
        print(f"  [警告] 车辆参数恢复失败 {npz_path}: {exc}")
        vehicle_ctx = {}

    try:
        rail = RailParams(rail_type=rail_type, yaml_dir=param_profile_dir)
        rail_ctx = _dataclass_to_dict(rail)
    except Exception as exc:
        print(f"  [警告] 钢轨参数恢复失败 {npz_path}: {exc}")
        rail_ctx = {}

    try:
        fastener = Fastener_KV(fastener_type=fastener_type, yaml_dir=param_profile_dir)
        fastener_ctx = _dataclass_to_dict(fastener)
    except Exception as exc:
        print(f"  [警告] 扣件参数恢复失败 {npz_path}: {exc}")
        fastener_ctx = {}

    try:
        subrail = Subrail_Params(subrail_type="Standard_Subrail", yaml_dir=param_profile_dir)
        subrail_ctx = _dataclass_to_dict(subrail)
    except Exception as exc:
        print(f"  [警告] 轨下结构参数恢复失败 {npz_path}: {exc}")
        subrail_ctx = {}

    line_ctx: Dict[str, Any] = {}
    try:
        track_alignment = _get_track_alignment(meta)
        (
            k1,
            h1,
            g1,
            Rcurve,
            Thetacurve,
            Lcurve,
            curvecase,
            L1,
            L2,
            Lz1,
            Lz2,
            hcg,
            S,
            ZH_abs,
        ) = track_alignment.get_geometry_at(start_mileage_m)
        line_ctx = {
            "start_mileage_m": start_mileage_m,
            "curvature_1pm": float(k1),
            "cant_rad": float(h1),
            "gradient": float(g1),
            "curve_radius_m": float(Rcurve),
            "curve_theta_rad": float(Thetacurve),
            "curve_length_m": float(Lcurve),
            "curve_case": int(curvecase),
            "transition_in_m": float(L1),
            "transition_out_m": float(L2),
            "straight_in_m": float(Lz1),
            "straight_out_m": float(Lz2),
            "design_cant_m": float(hcg),
            "rail_center_distance_m": float(S),
            "zh_abs_m": float(ZH_abs),
        }
    except Exception as exc:
        print(f"  [警告] 线型参数恢复失败 {npz_path}: {exc}")

    return {
        "run_metadata": meta,
        "vehicle": vehicle_ctx,
        "rail": rail_ctx,
        "fastener": fastener_ctx,
        "subrail": subrail_ctx,
        "line": line_ctx,
    }


class VTCMDataset(Dataset):
    """
    VTCM PINO 数据集类
    
    从多个仿真结果 NPZ 文件加载数据，并将其分割成固定长度的时间窗口，
    以及对应的输入激励和目标输出。
    """

    def __init__(
        self,
        npz_files: list,
        dof: int = 35,
        seq_length: int = 256,
        stride: int = 128,
        normalize: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            npz_files: 仿真结果 NPZ 文件的路径列表
            dof: 自由度数 (默认 35)
            seq_length: 时间窗口长度 (默认 256)
            stride: 滑动窗口步长 (默认 128，即 50% 重叠)
            normalize: 是否对数据进行归一化
            device: torch 设备 ('cpu' 或 'cuda')
        """
        self.npz_files = npz_files
        self.dof = dof
        self.seq_length = seq_length
        self.stride = stride
        self.normalize = normalize
        self.device = device

        self.data = []
        self.file_contexts: Dict[str, Dict[str, Any]] = {}
        self.norm_params = {"X_mean": None, "X_std": None, "u_mean": None, "u_std": None}

        self._load_all_data()
        # self._normalize()

    def _load_all_data(self) -> None:
        """从所有 NPZ 文件加载仿真数据"""
        print(f"加载 {len(self.npz_files)} 个 NPZ 文件...")
        
        for npz_path in self.npz_files:
            try:
                # 读取仿真结果和物理上下文，如果self.file_contexts中已有则复用
                data_dict = np.load(npz_path, allow_pickle=True)
                physics_context = self.file_contexts.setdefault(
                    str(npz_path), _build_physics_context(Path(npz_path))
                )
                X = data_dict.get("X")
                V = data_dict.get("V")
                A = data_dict.get("A")
                dt = float(data_dict.get("dt", 1e-4))

                if X is None or V is None or A is None:
                    print(f"  [跳过] {npz_path}: 缺少必要的 X/V/A 字段")
                    continue

                Nt = X.shape[0]
                if Nt < self.seq_length:
                    print(
                        f"  [跳过] {npz_path}: 长度 {Nt} < seq_length {self.seq_length}"
                    )
                    continue

                # 构建输入激励 u(t): 左右轨垂向不平顺分离为 2 通道
                irre_bz_L = data_dict.get("Irre_bz_L_ref")
                irre_bz_R = data_dict.get("Irre_bz_R_ref")

                if irre_bz_L is not None and irre_bz_R is not None:
                    uL = np.asarray(irre_bz_L[:Nt], dtype=np.float32)
                    uR = np.asarray(irre_bz_R[:Nt], dtype=np.float32)
                    u = np.stack([uL, uR], axis=1)  # [Nt, 2]
                elif irre_bz_L is not None:
                    uL = np.asarray(irre_bz_L[:Nt], dtype=np.float32)
                    u = np.stack([uL, np.zeros_like(uL)], axis=1)  # [Nt, 2]
                elif irre_bz_R is not None:
                    uR = np.asarray(irre_bz_R[:Nt], dtype=np.float32)
                    u = np.stack([np.zeros_like(uR), uR], axis=1)  # [Nt, 2]
                else:
                    u = np.zeros((Nt, 2), dtype=np.float32)

                # 构建目标输出 y(t) = [X; V; A]
                y = np.concatenate(
                    [X.reshape(Nt, -1), V.reshape(Nt, -1), A.reshape(Nt, -1)], axis=1
                )

                # 用滑动窗口分割数据，跳过初始的2秒
                skip_initial_seconds = 2.0  # 跳过初始2秒的无激励数据
                skip_steps = int(skip_initial_seconds / dt)
                
                windows_before = len(self.data)
                for start_idx in range(skip_steps, Nt - self.seq_length + 1, self.stride):
                    u_window = u[start_idx : start_idx + self.seq_length]
                    y_window = y[start_idx : start_idx + self.seq_length]

                    self.data.append({
                        "u": u_window.astype(np.float32),
                        "y": y_window.astype(np.float32),
                        "dt": dt,
                        "source_file": str(npz_path),
                        "window_start": int(start_idx),
                        "window_end": int(start_idx + self.seq_length),
                        "physics_context": physics_context,
                    })

                print(f"  [成功] {npz_path}: 加载 {len(self.data) - windows_before} 个窗口 (跳过初始 {skip_initial_seconds}s)")

            except Exception as e:
                print(f"  [错误] {npz_path}: {e}")

        print(f"总共加载 {len(self.data)} 个时间窗口")

    def _normalize(self) -> None:
        """计算并应用全局归一化参数"""
        if not self.normalize or len(self.data) == 0:
            return

        print("计算归一化参数...")
        u_all = np.concatenate([d["u"] for d in self.data], axis=0)
        y_all = np.concatenate([d["y"] for d in self.data], axis=0)

        # 输入按通道归一化（适配 [N*T, 2]）
        if u_all.ndim == 2:
            self.norm_params["u_mean"] = u_all.mean(axis=0, keepdims=True)      # [1, 2]
            self.norm_params["u_std"] = u_all.std(axis=0, keepdims=True) + 1e-8 # [1, 2]
        else:
            self.norm_params["u_mean"] = u_all.mean()
            self.norm_params["u_std"] = u_all.std() + 1e-8
        self.norm_params["X_mean"] = y_all.mean(axis=0, keepdims=True)
        self.norm_params["X_std"] = y_all.std(axis=0, keepdims=True) + 1e-8

        for d in self.data:
            d["u"] = (d["u"] - self.norm_params["u_mean"]) / self.norm_params["u_std"]
            d["y"] = (d["y"] - self.norm_params["X_mean"]) / self.norm_params["X_std"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # 转换为 torch 张量
        u = torch.from_numpy(sample["u"]).to(self.device)
        y = torch.from_numpy(sample["y"]).to(self.device)

        # u: [seq_length, 2] -> [2, seq_length]（左右轨双通道）
        # 兼容旧数据: [seq_length] -> [1, seq_length]
        # y: [seq_length, dof*3] -> [dof*3, seq_length] (转置为通道-时间格式)
        if u.dim() == 1:
            u = u.unsqueeze(0)  # [1, seq_length]
        elif u.dim() == 2:
            u = u.transpose(0, 1)  # [2, seq_length]
        else:
            raise ValueError(f"不支持的输入 u 维度: {tuple(u.shape)}")
        y = y.transpose(0, 1)  # [dof*3, seq_length]

        return {
            "input": u,
            "output": y,
            "dt": sample["dt"],
            "source_file": sample["source_file"],
            "window_start": sample["window_start"],
            "window_end": sample["window_end"],
            "physics_context": sample["physics_context"],
        }

    def denormalize_output(self, y_norm: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        反归一化模型输出 y。

        支持输入形状:
          - torch: [B, C, T] 或 [C, T]
          - numpy: [B, C, T] 或 [C, T]
        其中 C = dof * 3。
        """
        mean = self.norm_params.get("X_mean")
        std = self.norm_params.get("X_std")
        if mean is None or std is None:
            return y_norm

        if isinstance(y_norm, torch.Tensor):
            mean_t = torch.as_tensor(mean, dtype=y_norm.dtype, device=y_norm.device).squeeze(0)  # [C]
            std_t = torch.as_tensor(std, dtype=y_norm.dtype, device=y_norm.device).squeeze(0)    # [C]
            if y_norm.dim() == 3:      # [B, C, T]
                return y_norm * std_t.view(1, -1, 1) + mean_t.view(1, -1, 1)
            if y_norm.dim() == 2:      # [C, T]
                return y_norm * std_t.view(-1, 1) + mean_t.view(-1, 1)
            raise ValueError(f"不支持的 y_norm 维度: {tuple(y_norm.shape)}")

        y_arr = np.asarray(y_norm)
        mean_arr = np.asarray(mean, dtype=y_arr.dtype).reshape(-1)   # [C]
        std_arr = np.asarray(std, dtype=y_arr.dtype).reshape(-1)     # [C]
        if y_arr.ndim == 3:          # [B, C, T]
            return y_arr * std_arr.reshape(1, -1, 1) + mean_arr.reshape(1, -1, 1)
        if y_arr.ndim == 2:          # [C, T]
            return y_arr * std_arr.reshape(-1, 1) + mean_arr.reshape(-1, 1)
        raise ValueError(f"不支持的 y_norm 维度: {y_arr.shape}")

    def denormalize_input(self, u_norm: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """反归一化输入 u，支持 [B,C,T]/[C,T]/[T]，其中 C 可为 1 或 2。"""
        mean = self.norm_params.get("u_mean")
        std = self.norm_params.get("u_std")
        if mean is None or std is None:
            return u_norm

        if isinstance(u_norm, torch.Tensor):
            mean_t = torch.as_tensor(mean, dtype=u_norm.dtype, device=u_norm.device)
            std_t = torch.as_tensor(std, dtype=u_norm.dtype, device=u_norm.device)
            return u_norm * std_t + mean_t

        u_arr = np.asarray(u_norm)
        return u_arr * np.asarray(std, dtype=u_arr.dtype) + np.asarray(mean, dtype=u_arr.dtype)


def create_dataloader(
    npz_dir: str,
    batch_size: int = 4,
    seq_length: int = 256,
    stride: int = 128,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    从目录中加载所有 NPZ 文件并创建训练/验证数据加载器
    
    Args:
        npz_dir: 包含仿真结果的目录路径 (e.g., 'results/default_project')
        batch_size: 批大小
        seq_length: 时间窗口长度
        stride: 滑动窗口步长
        train_ratio: 训练集比例 (仅当使用单一项目时)
        shuffle: 是否随机打乱
        device: torch 设备
    
    Returns:
        (train_dataloader, val_dataloader, norm_params)
    """
    npz_dir = Path(npz_dir)
    
    # 递归查找所有 simulation_result.npz 文件
    npz_files = list(npz_dir.rglob("simulation_result.npz"))
    
    if len(npz_files) == 0:
        raise ValueError(f"未找到任何 NPZ 文件，请检查目录: {npz_dir}")
    
    print(f"发现 {len(npz_files)} 个仿真结果文件")
    
    # 创建数据集
    dataset = VTCMDataset(
        npz_files=npz_files,
        seq_length=seq_length,
        stride=stride,
        normalize=True,
        device=device,
    )
    
    # 分割训练/验证集
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"训练集: {len(train_dataset)} 个样本 ({n_train} 个窗口)")
    print(f"验证集: {len(val_dataset)} 个样本 ({n_val} 个窗口)")
    
    return train_loader, val_loader, dataset.norm_params


def create_train_test_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 4,
    seq_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    分别从训练和测试目录加载数据，并创建数据加载器
    
    使用训练集的归一化参数对测试集进行归一化，确保一致性。
    
    Args:
        train_dir: 训练集结果目录路径
        test_dir: 测试集结果目录路径
        batch_size: 批大小
        seq_length: 时间窗口长度
        stride: 滑动窗口步长
        shuffle: 训练集是否随机打乱
        device: torch 设备
    
    Returns:
        (train_dataloader, test_dataloader, norm_params)
    """
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    
    # 查找训练集文件
    train_npz_files = list(train_dir.rglob("simulation_result.npz"))
    if len(train_npz_files) == 0:
        raise ValueError(f"未找到训练集 NPZ 文件: {train_dir}")
    
    # 查找测试集文件
    test_npz_files = list(test_dir.rglob("simulation_result.npz"))
    if len(test_npz_files) == 0:
        raise ValueError(f"未找到测试集 NPZ 文件: {test_dir}")
    
    print(f"发现 {len(train_npz_files)} 个训练集仿真结果")
    print(f"发现 {len(test_npz_files)} 个测试集仿真结果")
    
    # 加载训练集
    print("\n📊 加载训练集...")
    train_dataset = VTCMDataset(
        npz_files=train_npz_files,
        seq_length=seq_length,
        stride=stride,
        normalize=True,
        device=device,
    )
    
    # 加载测试集并使用训练集的归一化参数
    print("\n📊 加载测试集...")
    test_dataset = VTCMDataset(
        npz_files=test_npz_files,
        seq_length=seq_length,
        stride=stride,
        normalize=False,  # 先不归一化
        device=device,
    )
    
    # 用训练集的参数对测试集进行归一化
    if len(train_dataset.data) > 0:
        norm_params = train_dataset.norm_params
        for d in test_dataset.data:
            d["u"] = (d["u"] - norm_params["u_mean"]) / norm_params["u_std"]
            d["y"] = (d["y"] - norm_params["X_mean"]) / norm_params["X_std"]
    else:
        norm_params = test_dataset.norm_params
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"\n✓ 数据加载完成")
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  测试集: {len(test_dataset)} 个样本")
    
    return train_loader, test_loader, norm_params


if __name__ == "__main__":
    # 示例：加载数据集
    train_loader, val_loader, norm_params = create_dataloader(
        npz_dir="results/default_project",
        batch_size=2,
        seq_length=256,
        stride=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print("\n示例批次:")
    for batch_idx, batch in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"  input shape: {batch['input'].shape}")
        print(f"  output shape: {batch['output'].shape}")
        if batch_idx >= 2:
            break
