"""
PINO模型推理工具 - 集成norm_stats的完整工作流

功能：
- 加载已训练的模型和归一化参数
- 对新数据进行推理（自动处理标准化/反标准化）
- 支持单个样本和批量推理
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


def load_norm_stats(norm_file_path: Union[str, Path] = "Dataset/norm_stats.npz") -> Dict[str, np.ndarray]:
    """
    加载归一化参数文件
    
    参数:
        norm_file_path: 指向 norm_stats.npz 文件的路径
    
    返回:
        dict: 包含 'z_mu', 'z_sigma', 'ds' 的字典
    
    例子:
        >>> norm_stats = load_norm_stats("Dataset/norm_stats.npz")
        >>> print(f"特征数: {norm_stats['z_mu'].shape[1]}")
    """
    npz_file = Path(norm_file_path)
    if not npz_file.exists():
        raise FileNotFoundError(f"找不到文件: {npz_file.absolute()}")
    
    data = np.load(npz_file)
    return {
        'z_mu': data['z_mu'],
        'z_sigma': data['z_sigma'],
        'ds': data['ds'],
    }


def normalize_data(
    z_raw: np.ndarray, 
    norm_stats: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    对原始数据进行标准化 (Z-score normalization)
    
    公式: z_norm = (z_raw - z_mu) / z_sigma
    
    参数:
        z_raw: 原始数据，支持形状 (n_samples, n_features)、(n_features,) 或 (batch, time, features)
        norm_stats: load_norm_stats() 返回的字典
    
    返回:
        z_norm: 标准化后的数据（与输入形状相同）
    
    例子:
        >>> z_raw = np.random.randn(10, 105)
        >>> z_norm = normalize_data(z_raw, norm_stats)
        >>> print(f"标准化后的均值: {z_norm.mean():.6f}")  # 应接近0
    """
    z_mu = norm_stats['z_mu']
    z_sigma = norm_stats['z_sigma']
    return (z_raw - z_mu) / z_sigma


def denormalize_data(
    z_norm: np.ndarray, 
    norm_stats: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    反标准化数据（将标准化数据转换回原始尺度）
    
    公式: z_raw = z_norm * z_sigma + z_mu
    
    参数:
        z_norm: 标准化后的数据
        norm_stats: load_norm_stats() 返回的字典
    
    返回:
        z_raw: 原始尺度的数据
    
    例子:
        >>> z_norm = normalize_data(z_raw, norm_stats)
        >>> z_recovered = denormalize_data(z_norm, norm_stats)
        >>> assert np.allclose(z_raw, z_recovered)
    """
    z_mu = norm_stats['z_mu']
    z_sigma = norm_stats['z_sigma']
    return z_norm * z_sigma + z_mu


class PINOInference:
    """
    PINO模型推理管理器
    
    功能：
    - 自动处理模型和参数加载
    - 包装推理过程中的标准化/反标准化
    - 支持多种推理模式
    
    例子:
        >>> inference = PINOInference("model.pt", "Dataset/norm_stats.npz")
        >>> prediction = inference(raw_input_data)
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        norm_stats_path: Union[str, Path] = "Dataset/norm_stats.npz",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化推理管理器
        
        参数:
            model_path: 保存的模型路径
            norm_stats_path: 归一化参数文件路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.model_path = Path(model_path)
        self.norm_stats_path = Path(norm_stats_path)
        
        # 加载归一化参数
        self.norm_stats = load_norm_stats(self.norm_stats_path)
        print(f"✓ 已加载norm_stats: {self.norm_stats_path}")
        
        # 加载模型（需要外部定义模型架构）
        # self.model 应在子类或外部设置
        self.model = None
        print(f"✓ 已初始化推理管理器（device={self.device}）")
    
    def preprocess(self, z_raw: np.ndarray) -> torch.Tensor:
        """
        预处理：原始数据 → 标准化 → Tensor
        
        参数:
            z_raw: 原始数据 (numpy array)
        
        返回:
            标准化后的PyTorch Tensor
        """
        z_normalized = normalize_data(z_raw, self.norm_stats)
        return torch.from_numpy(z_normalized).float().to(self.device)
    
    def postprocess(self, z_norm_tensor: torch.Tensor) -> np.ndarray:
        """
        后处理：标准化Tensor → 反标准化 → 原始数据
        
        参数:
            z_norm_tensor: 标准化后的PyTorch Tensor
        
        返回:
            原始尺度的numpy数据
        """
        z_norm_np = z_norm_tensor.detach().cpu().numpy()
        return denormalize_data(z_norm_np, self.norm_stats)
    
    def inference(self, z_raw: np.ndarray) -> np.ndarray:
        """
        完整推理流程：原始输入 → 模型推理 → 原始输出
        
        参数:
            z_raw: 原始数据
        
        返回:
            原始尺度的预测结果
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先设置 self.model")
        
        # 预处理
        z_in = self.preprocess(z_raw)
        
        # 推理
        self.model.eval()
        with torch.no_grad():
            z_out_norm = self.model(z_in)
        
        # 后处理
        return self.postprocess(z_out_norm)


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    norm_stats: Dict[str, np.ndarray],
    checkpoint_path: Union[str, Path] = "pino_checkpoint.pt",
    epoch: int = 0,
    loss: float = 0.0,
) -> None:
    """
    保存模型和归一化参数为单一检查点
    
    参数:
        model: PyTorch模型
        optimizer: 优化器
        norm_stats: 归一化参数字典
        checkpoint_path: 保存路径
        epoch: 当前轮数（用于记录）
        loss: 当前损失（用于记录）
    
    例子:
        >>> save_model_checkpoint(model, optimizer, norm_stats, "checkpoints/epoch_100.pt")
    """
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'z_mu': norm_stats['z_mu'],
        'z_sigma': norm_stats['z_sigma'],
        'ds': norm_stats['ds'],
    }
    
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ 已保存检查点到: {checkpoint_path} (epoch={epoch})")


def load_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Union[str, Path] = "pino_checkpoint.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, np.ndarray], Dict]:
    """
    加载模型、优化器和归一化参数
    
    参数:
        model: PyTorch模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        device: 计算设备
    
    返回:
        (model, optimizer, norm_stats, metadata) 其中metadata包含 epoch 和 loss
    
    例子:
        >>> model, opt, norm_stats, meta = load_model_checkpoint(model, optimizer)
        >>> print(f"从第 {meta['epoch']} 轮继续训练")
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    norm_stats = {
        'z_mu': checkpoint['z_mu'],
        'z_sigma': checkpoint['z_sigma'],
        'ds': checkpoint['ds'],
    }
    
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
    }
    
    print(f"✓ 已加载检查点: {checkpoint_path}")
    print(f"  - 轮数: {metadata['epoch']}, 损失: {metadata['loss']:.6f}")
    
    return model, optimizer, norm_stats, metadata


# ============================================================================
# 推理示例
# ============================================================================

def example_inference():
    """
    推理示例代码
    
    使用场景：对新的原始数据进行推理，获得原始尺度的预测结果
    """
    # 1. 加载归一化参数
    norm_stats = load_norm_stats("Dataset/norm_stats.npz")
    print(f"✓ 加载norm_stats:")
    print(f"  - z_mu 形状: {norm_stats['z_mu'].shape}")
    print(f"  - z_sigma 形状: {norm_stats['z_sigma'].shape}")
    
    # 2. 创建模拟的原始数据
    z_raw = np.random.randn(10, 105).astype(np.float32)  # (时间步, 特征)
    print(f"\n✓ 创建测试数据:")
    print(f"  - 形状: {z_raw.shape}")
    print(f"  - 数值范围: [{z_raw.min():.4f}, {z_raw.max():.4f}]")
    
    # 3. 标准化
    z_normalized = normalize_data(z_raw, norm_stats)
    print(f"\n✓ 标准化后:")
    print(f"  - 均值: {z_normalized.mean():.8f} (应接近0)")
    print(f"  - 标准差: {z_normalized.std():.8f} (应接近1)")
    
    # 4. 模型推理 (这里用标准化数据代替)
    z_pred_norm = z_normalized.copy()  # 实际应用中用模型推理替换
    
    # 5. 反标准化得到原始尺度结果
    z_pred = denormalize_data(z_pred_norm, norm_stats)
    print(f"\n✓ 反标准化后:")
    print(f"  - 形状: {z_pred.shape}")
    print(f"  - 数值范围: [{z_pred.min():.4f}, {z_pred.max():.4f}]")
    
    # 6. 验证
    error = np.abs(z_raw - z_pred).max()
    print(f"\n✓ 验证 (原始vs恢复):")
    print(f"  - 最大误差: {error:.8f}")
    print(f"  - 恢复成功: {np.allclose(z_raw, z_pred, atol=1e-5)}")


if __name__ == "__main__":
    example_inference()
