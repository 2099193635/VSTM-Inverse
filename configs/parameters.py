import numpy as np
from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
import os
import random
try:
    import yaml
except Exception:
    yaml = None


def _resolve_profile_dir(profile_dir: str) -> str:
    """将参数目录解析为绝对路径；相对路径默认相对于项目根目录。"""
    if os.path.isabs(profile_dir):
        return profile_dir
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.abspath(os.path.join(project_root, profile_dir))


def _load_yaml_with_fallback(profile_dir: str, yaml_name: str, fallback_dir: str = 'configs/standard'):
    """
    读取 YAML：先读 profile_dir，再回退到 fallback_dir。
    返回 dict；若两处都不存在返回空 dict。
    """
    candidates = [
        os.path.join(_resolve_profile_dir(profile_dir), yaml_name),
        os.path.join(_resolve_profile_dir(fallback_dir), yaml_name),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        if yaml is None:
            raise ImportError(f"检测到 YAML 文件但未安装 PyYAML: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML 顶层结构必须为字典: {path}")
        return data

    return {}


def _deep_update(base: dict, updates: dict) -> dict:
    """递归更新字典，返回更新后的 base。"""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

class CurveFunction:
    """
    曲线参数连续函数计算类 (替代 MATLAB 的符号匿名函数)
    利用解析解直接计算曲率半径、超高角及其各阶导数，支持向量化计算，求解极速。
    """
    def __init__(self, Rcurve: float, Thetacurve: float, curvecase: int, 
                 L1: float, L2: float, Lz1: float, Lz2: float, hcg: float, S: float, v: float):
        self.Rcurve = Rcurve
        self.Thetacurve = np.abs(Thetacurve)
        self.curvecase = curvecase
        self.L1 = L1
        self.L2 = L2
        self.Lz1 = Lz1
        self.Lz2 = Lz2
        self.hcg = hcg
        self.S = S
        self.v = v

        # --- 预计算基本参数 ---
        self.Lcurve = self.Thetacurve * Rcurve          # 圆曲线长度
        self.thetayy = np.arctan(hcg / S)          # 圆曲线上超高角(右转方向幅值)
        self.K0 = 1.0 / Rcurve if Rcurve != 0 else 0.0 # 圆曲线曲率
        
        # --- 预计算里程分界节点 ---
        self.p1 = Lz1                              # 入缓点 (ZH)
        self.p2 = Lz1 + L1                         # 缓圆点 (HY)
        self.p3 = self.p2 + self.Lcurve            # 圆缓点 (YH)
        self.p4 = self.p3 + L2                     # 缓直点 (HZ)
        self.p5 = self.p4 + Lz2                    # 终点
        
        # --- 超高角翻转标志 (适配原代码: curvecase==2 时翻转) ---
        self.dir_sign = -1.0 if curvecase == 2 else 1.0

    def R_mile(self, x):
        """当前里程 x 处的曲率半径"""
        x = np.asarray(x)
        R = np.full_like(x, 1e10, dtype=float)  # 默认直线段曲率半径极大
        
        # 1. 前缓和曲线 (ZH - HY)
        m1 = (x >= self.p1) & (x < self.p2)
        if self.L1 > 0:
            # 避免除以 0，加一个极小值 1e-10
            R[m1] = 1.0 / (self.K0 * (x[m1] - self.p1) / self.L1 + 1e-10)
            
        # 2. 圆曲线 (HY - YH)
        m2 = (x >= self.p2) & (x < self.p3)
        R[m2] = self.Rcurve
        
        # 3. 后缓和曲线 (YH - HZ)
        m3 = (x >= self.p3) & (x < self.p4)
        if self.L2 > 0:
            R[m3] = 1.0 / (self.K0 * (self.p4 - x[m3]) / self.L2 + 1e-10)
            
        return R[()] if R.ndim == 0 else R

    def Theta_mile(self, x):
        """当前里程 x 处的超高角 (rad)"""
        x = np.asarray(x)
        Theta = np.zeros_like(x, dtype=float)
        
        if self.L1 > 0:
            m1 = (x >= self.p1) & (x < self.p2)
            Theta[m1] = self.thetayy * (x[m1] - self.p1) / self.L1
            
        m2 = (x >= self.p2) & (x < self.p3)
        Theta[m2] = self.thetayy
        
        if self.L2 > 0:
            m3 = (x >= self.p3) & (x < self.p4)
            Theta[m3] = self.thetayy * (self.p4 - x[m3]) / self.L2
            
        Theta = Theta * self.dir_sign
        return Theta[()] if Theta.ndim == 0 else Theta

    def dTheta_mile(self, x):
        """当前里程 x 处的超高角一阶变化率 (dTheta/dt)"""
        x = np.asarray(x)
        dTheta = np.zeros_like(x, dtype=float)
        
        # 缓和曲线段：dTheta/dt = (dTheta/dx) * (dx/dt) = (thetayy / L) * v
        if self.L1 > 0:
            m1 = (x >= self.p1) & (x < self.p2)
            dTheta[m1] = (self.thetayy / self.L1) * self.v
            
        if self.L2 > 0:
            m3 = (x >= self.p3) & (x < self.p4)
            dTheta[m3] = -(self.thetayy / self.L2) * self.v
            
        dTheta = dTheta * self.dir_sign
        return dTheta[()] if dTheta.ndim == 0 else dTheta

    def ddTheta_mile(self, x):
        """当前里程 x 处的超高角二阶变化率 (ddTheta/dt^2)"""
        # 由于所有段的超高要么是常数，要么是关于里程的线性函数，
        # 在恒定车速 v 下，所有区间内部的二阶导数恒为 0。
        x = np.asarray(x)
        return np.zeros_like(x, dtype=float)[()] if x.ndim == 0 else np.zeros_like(x, dtype=float)

    def dK_mile(self, x):
        """当前里程 x 处的曲率一阶变化率 (dK/dt)"""
        x = np.asarray(x)
        dK = np.zeros_like(x, dtype=float)
        
        # dK/dt = (dK/dx) * v = (K0 / L) * v
        if self.L1 > 0:
            m1 = (x >= self.p1) & (x < self.p2)
            dK[m1] = (self.K0 / self.L1) * self.v
            
        if self.L2 > 0:
            m3 = (x >= self.p3) & (x < self.p4)
            dK[m3] = -(self.K0 / self.L2) * self.v
            
        return dK[()] if dK.ndim == 0 else dK
    

@dataclass
class ExtraforceElementSwitch:
    """
    额外力元开关配置类
    """
    # 功能1. 额外减振力元按钮
    Switch_ExtraForceElement: str = 'Off'
    # 功能2. 曲线轨道按钮
    Switch_CurveTrack: str = 'Off'
    # 功能3. 胶垫纵向/横向分区按钮
    Switch_PadZone: str = 'Off'
    # 功能4. 两点接触按钮
    Switch_2PointContact: str = 'Off'
    # 功能5. 垫板分区按钮 
    Switch_PadPartition: str = 'Off'
    # 功能6. 非对称轨底坡按钮 
    Switch_RailCant_Unsymmetric: str = 'Off'
    def is_active(self, switch_name: str) -> bool:
        """
        辅助判断方法：将 'On'/'Off' 字符串转换为 Python 布尔值
        忽略大小写，防止误写成 'on' 或 'ON' 导致判断失败
        """
        # 获取对应的属性值，如果属性不存在则默认返回 'Off'
        val = getattr(self, switch_name, 'Off')
        return str(val).lower() == 'on'

@dataclass
class IntegrationParams:
    """
    /控制模块/ - 设置数值积分运行参量
    处理仿真时间、网格划分、车速以及相对/绝对坐标系的映射
    """
    # 1. 外部依赖参量 
    Lc: float = 9.0                 # 车辆半轴距 (m) - 从车辆参数注入
    Lt: float = 1.25                # 转向架半轴距 (m) - 从车辆参数注入
    R: float = 0.46                 # 车轮半径 (m) - 从车辆参数注入
    Lkj: float = 0.6                # 扣件间距 (m) - 从扣件参数注入
    # 2. 仿真宏观控制参量
    Vx_set: float = None            # 指定车速(km/h)
    Tz: float = 5.0                 # 期望仿真总时间
    Tstep: float = 1e-4             # 积分步长
    # 3. 空间坐标映射参量
    S0_mileage: float = 801000.0    # 绝对坐标：仿真起点对应的真实线路绝对里程 (m)
    Nsub: int = 2000                # 相对坐标：轨道子结构(扣件)数量
    X0: float = 20.0                # 相对坐标：第四轮对在柔性轨道上的初始位置 (m)
    # ==========================================
    # 4. 求解器算法参量 (Newmark-β / Zhai方法等)
    # ==========================================
    alpha: float = 0.5          # 数值积分控制参数
    beta: float = 0.5           # 数值积分控制参数
    g: float = 9.81             # 重力加速度 (m/s^2)
    # 内部推导参量 (不可通过构造函数传入，由程序自动计算)
    Vx: float = field(init=False)
    Vc: float = field(init=False)
    omega: float = field(init=False)
    Ljs: float = field(init=False)
    Lz: float = field(init=False)
    Cord_fastener: np.ndarray = field(init=False)
    Nt: int = field(init=False)

    def __post_init__(self):
        # 1. 运动学参量计算
        if self.Vx_set is not None:
            self.Vx = self.Vx_set
        else:
            # 随机生成 250 ~ 350 km/h 之间的车速，保留两位小数
            self.Vx = round(250.0 + random.random() * 100.0, 2)
        self.Vc = self.Vx / 3.6             # 车辆运行速度 (m/s)
        self.omega = self.Vc / self.R       # 车轮运行转动速度 (rad/s)
        # 2. 相对柔性轨道边界计算与防脱轨保护
        self.Ljs = self.Lkj * self.Nsub     # 柔性钢轨计算总长度 (m)
        self.Lz = self.Tz * self.Vc         # 理论期望走行距离 (m)
        # 检算：当前车速下，期望行驶距离是否超出了轨道末端
        # Ljs(总长) - X0 - 2*(Lt+Lc)
        max_Lz = self.Ljs - self.X0 - 2 * (self.Lt + self.Lc)
        if self.Lz > max_Lz:
            print(f"[警告] 期望走行距离 {self.Lz:.2f}m 超出了柔性轨道边界！")
            print(f"       为防止车辆脱轨报错，已自动将走行距离截断为 {max_Lz:.2f}m。")
            self.Lz = max_Lz
            self.Tz = self.Lz / self.Vc     # 同步缩短仿真总时间
        # 3. 空间网格与时域网格划分
        # a. 生成相对坐标系下的全钢轨扣件纵向坐标网格 (0 ~ Ljs)
        # 采用 np.linspace 避免浮点累加造成的末尾截断误差
        self.Cord_fastener = np.linspace(0, self.Nsub * self.Lkj, self.Nsub + 1)
        # b. 计算总积分步数
        self.Nt = int(round(self.Tz / self.Tstep)) 

    def get_absolute_mileage(self, t: float) -> float:
        """
        核心映射方法：根据当前仿真时间 t (s)，返回车辆所在的真实绝对里程 s (m)
        用于向 RealTrackAlignment 获取实时曲率、超高等参数
        """
        return self.S0_mileage + self.X0 + self.Vc * t
    
@dataclass
class ModesParameters:
    """钢轨模态求解参数配置类"""
    def __init__(self, 
                 NV: int = 350,
                 NL: int = 350,
                 NT: int = 350):
        self.NV: int = NV  # 钢轨垂向位移模态
        self.NL: int = NL  # 钢轨横向位移模态
        self.NT: int = NT  # 钢轨扭转位移模态


@dataclass
class RealTrackAlignment:
    """
    解析线路参数表，生成全线参数矩阵
    """
    def __init__(self, 
                 curve_file_dir: str = 'configs\curve_parameters.csv', 
                 gradient_file_dir: str = 'configs\gradient_parameters.csv',
                 cache_file_dir: str = 'configs/track_cache.npz',
                 force_rebuild: bool = False):
        self.curve_file_dir = curve_file_dir
        self.gradient_file_dir = gradient_file_dir
        self.cache_file_dir = cache_file_dir
        self.ds = 0.1           # 离散步长
        self.S_val = 1.5099     # 钢轨扭心距常量
        if os.path.exists(self.cache_file_dir) and not force_rebuild:
            print(f"检测到轨道参数缓存文件，正在加载: {self.cache_file_dir}")
            data = np.load(self.cache_file_dir)
            self.s_grid = data['s_grid']
            self.k_grid, self.h_grid, self.g_grid = data['k_grid'], data['h_grid'], data['g_grid']
            self.R_grid, self.Theta_grid, self.Lc_grid = data['R_grid'], data['Theta_grid'], data['Lc_grid']
            self.case_grid, self.L1_grid, self.L2_grid = data['case_grid'], data['L1_grid'], data['L2_grid']
            self.Lz1_grid, self.Lz2_grid, self.hcg_grid = data['Lz1_grid'], data['Lz2_grid'], data['hcg_grid']
            self.S_grid = data['S_grid']
            self.ZH_grid = data['ZH_grid']
        else:
            if force_rebuild:
                print(f" -> [强制重建模式] 忽略已有缓存，正在重新解析 CSV 原始数据...")
            else:
                print(f" -> [初次运行] 未找到缓存文件，正在解析原始数据并生成全线参数矩阵...")
            self.curve_data = pd.read_csv(self.curve_file_dir)
            self.gradient_data = pd.read_csv(self.gradient_file_dir)
            min_s = min(self.curve_data['Start'].min(), self.gradient_data['Start'].min()) * 1000
            max_s = max(self.curve_data['End'].max(), self.gradient_data['End'].max()) * 1000

            self.s_grid = np.arange(min_s, max_s, 0.1)
            self.k_grid = np.zeros_like(self.s_grid)
            self.h_grid = np.zeros_like(self.s_grid)
            self.g_grid = np.zeros_like(self.s_grid)

            self.R_grid = np.zeros_like(self.s_grid)
            self.Theta_grid = np.zeros_like(self.s_grid)
            self.Lc_grid = np.zeros_like(self.s_grid)
            self.case_grid = np.zeros_like(self.s_grid)
            self.L1_grid = np.zeros_like(self.s_grid)
            self.L2_grid = np.zeros_like(self.s_grid)
            self.Lz1_grid = np.zeros_like(self.s_grid)
            self.Lz2_grid = np.zeros_like(self.s_grid)
            self.hcg_grid = np.zeros_like(self.s_grid)
            self.S_grid = np.full_like(self.s_grid, self.S_val)
            self.ZH_grid = np.zeros_like(self.s_grid)

            self._parse_curves()
            self._parse_gradients()

            os.makedirs(os.path.dirname(self.cache_file_dir), exist_ok=True)
            np.savez_compressed(
                self.cache_file_dir, 
                s_grid=self.s_grid, k_grid=self.k_grid, h_grid=self.h_grid, g_grid=self.g_grid,
                R_grid=self.R_grid, Theta_grid=self.Theta_grid, Lc_grid=self.Lc_grid, case_grid=self.case_grid,
                L1_grid=self.L1_grid, L2_grid=self.L2_grid, Lz1_grid=self.Lz1_grid, Lz2_grid=self.Lz2_grid,
                hcg_grid=self.hcg_grid, S_grid=self.S_grid, ZH_grid=self.ZH_grid
            )


    def _parse_curves(self):
        """解析曲线表并叠加到全局网格"""
        num_curves = len(self.curve_data)
        for i, row in self.curve_data.iterrows():
            ZH = row['Start'] * 1000                        # 直缓点
            HZ = row['End'] * 1000                          # 缓直点
            L1 = row['Initial Transition Length']
            L2 = row['Final Transition Length']
            HY = ZH + L1                                    # 缓圆点
            YH = HZ - L2                                    # 圆缓点    

            R_val = row['Curve Radius']

            # 左转为正曲率，右转为负曲率
            dir_sign = 1 if row['Curve Direction'] == 'Left' else -1
            # 曲率计算：半径不为0时才计算曲率，否则保持为0（直线段）
            target_k = dir_sign * (1 / row['Curve Radius']) if row['Curve Radius'] != 0 else 0
            # 超高
            target_h = row['Superelevation'] * 0.001  

            # 计算宏观参数
            Lcurve = YH - HY
            Thetacurve = Lcurve / R_val if R_val != 0 else 0  # 曲线转角
            curvecase = dir_sign if R_val != 0 else 0  # 曲线类型

            # 计算夹直线 Lz1
            if i == 0:
                Lz1 = 500
            else:
                prev_HZ = self.curve_data.loc[i-1, 'End'] * 1000
                Lz1 = ZH - prev_HZ  
            
            # 计算夹直线 Lz2
            if i == num_curves - 1:
                Lz2 = 500
            else:
                next_ZH = self.curve_data.loc[i+1, 'Start'] * 1000
                Lz2 = next_ZH - HZ
            
            # =====================宏观参数计算======================
            curve_mask = (self.s_grid >= ZH) & (self.s_grid <= HZ)

            self.R_grid[curve_mask] = R_val
            self.Theta_grid[curve_mask] = Thetacurve
            self.Lc_grid[curve_mask] = Lcurve
            self.case_grid[curve_mask] = curvecase
            self.L1_grid[curve_mask] = L1
            self.L2_grid[curve_mask] = L2
            self.Lz1_grid[curve_mask] = Lz1
            self.Lz2_grid[curve_mask] = Lz2
            self.hcg_grid[curve_mask] = target_h
            self.ZH_grid[curve_mask] = ZH

            # ===============1. 前缓和曲线段：线性过渡==============
            mask_trans1 = (self.s_grid > ZH) & (self.s_grid <= HY)
            if L1 > 0:
                self.k_grid[mask_trans1] = target_k * (self.s_grid[mask_trans1] - ZH) / L1
                self.h_grid[mask_trans1] = target_h * (self.s_grid[mask_trans1] - ZH) / L1

            # ===============2. 圆曲线段：常数曲率和超高==============
            mask_circular = (self.s_grid > HY) & (self.s_grid <= YH)
            self.k_grid[mask_circular] = target_k
            self.h_grid[mask_circular] = target_h

            # ===============3. 后缓和曲线段：线性过渡回直线==============
            mask_trans2 = (self.s_grid > YH) & (self.s_grid <= HZ)
            if L2 > 0:
                self.k_grid[mask_trans2] = target_k * (HZ - self.s_grid[mask_trans2]) / L2
                self.h_grid[mask_trans2] = target_h * (HZ - self.s_grid[mask_trans2]) / L2
    
    def _parse_gradients(self):
        """解析坡度表并叠加到全局网格"""
        for _, row in self.gradient_data.iterrows():
            start = row['Start'] * 1000
            end = row['End'] * 1000
            gradient = row['Gradient'] * 0.001  # 转换为小数形式
            mask = (self.s_grid >= start) & (self.s_grid <= end)
            self.g_grid[mask] = gradient

    def get_geometry_at(self, s: float):
        """
        供动力学求解器调用：
        输入绝对里程 s (m)
        返回 ( 
        曲线半径 Rcurve, 
        曲线转角 Thetacurve, 
        圆曲线长度 Lcurve,
        曲线方向 curvecase,    # 1:左转, -1:右转, 0:直线
        前缓和曲线长度 L1,
        后缓和曲线长度 L2,
        前直线长度 L1,
        后直线长度 L2,
        设计超高值 hcg,
        钢轨两侧扭转中心距离 S
        )
        """
        s_min = self.s_grid[0]
        s_max = self.s_grid[-1]
        if s <= s_min:
            idx = 0
        elif s >= s_max:
            idx = len(self.s_grid) - 1
        else:
            idx = int((s - s_min) / self.ds + 0.5)

        return (
            # --- 1. 微观瞬态参数 ---
            self.k_grid[idx], self.h_grid[idx], self.g_grid[idx],
            # --- 2. 宏观线路参数 ---
            self.R_grid[idx],
            self.Theta_grid[idx],
            self.Lc_grid[idx],
            self.case_grid[idx],  # 直线段时这里会自动返回 0
            self.L1_grid[idx],
            self.L2_grid[idx],
            self.Lz1_grid[idx],
            self.Lz2_grid[idx],
            self.hcg_grid[idx],
            self.S_grid[idx],
            self.ZH_grid[idx]
        )
    
@dataclass
class Subrail_Params():
    """
    轨下结构参数配置
    """
    subrail_type: str = 'Standard_Subrail'  # 默认使用标准轨下参数
    yaml_dir: str = 'configs/standard'
    _PRESETS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Standard_Subrail': {
            'Ms': 237.0,                 # 轨枕质量 (kg)
            'Ls': 2.5,                   # 轨枕长度 (m)
            'd': 0.95,                   # 轨枕左右支点间距之半 (m)
            'Js': 237.0 * (2.5**2) / 12, # 轨枕侧滚惯量 (kg.m^2)
            'Mb': 682.6 / 2,             # 道床块质量 (kg)
            'Kbh': 5e7,                  # 轨枕横向刚度 (N/m)
            'Cbh': 4e4,                  # 轨枕横向阻尼 (N.s/m)
            'Kbv': 2.4e8,                # 道床块垂向刚度 (N/m)
            'Cbv': 5.88e4,               # 道床块垂向阻尼 (N.s/m)
            'Kw': 7.8e7,                 # 道床块剪切刚度 (N/m)
            'Cw': 8e4,                   # 道床块剪切阻尼 (N.s/m)
            'Kfv': 6.5e7,                # 路基刚度 (N/m)
            'Cfv': 3.1e4                 # 路基阻尼 (N.s/m)
        },

    }, repr=False)

    # 声明类属性，init=False 表示不通过构造函数直接传入
    Ms: float = field(init=False)
    Ls: float = field(init=False)
    d: float = field(init=False)
    Js: float = field(init=False)
    Mb: float = field(init=False)
    Kbh: float = field(init=False)
    Cbh: float = field(init=False)
    Kbv: float = field(init=False)
    Cbv: float = field(init=False)
    Kw: float = field(init=False)
    Cw: float = field(init=False)
    Kfv: float = field(init=False)
    Cfv: float = field(init=False)

    def __post_init__(self):
        y = _load_yaml_with_fallback(self.yaml_dir, 'subrail_params.yaml')
        if y:
            self._PRESETS = _deep_update(dict(self._PRESETS), y)
        self.reset_to_base()

    def reset_to_base(self):
        """定位轨下结构类型并注入属性"""
        if self.subrail_type not in self._PRESETS:
            raise ValueError(f"在轨下结构参数库中未找到类型: '{self.subrail_type}'，请检查输入或扩展字典。")

        conf = self._PRESETS[self.subrail_type]
        for key, value in conf.items():
            setattr(self, key, value)


@dataclass
class Fastener_KV():
    """
    KV模型扣件参数配置
    """
    fastener_type: str = 'Standard_KV'  # 默认使用标准KV扣件参数
    yaml_dir: str = 'configs/standard'
    _PRESETS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Standard_KV': {
            'Lkj': 0.6,       # 扣件间距 (m)
            'Kkjv': 60e6,     # 扣件垂向刚度 (N/m)
            'Ckjv': 50e3,     # 扣件垂向阻尼 (N.s/m)
            'Kkjh': 30e6,     # 扣件横向刚度 (N/m)
            'Ckjh': 50e3      # 扣件横向阻尼 (N.s/m)
        },
        # 可扩展
    }, repr=False)

    Lkj: float = field(init=False)
    Kkjv: float = field(init=False)
    Ckjv: float = field(init=False)
    Kkjh: float = field(init=False)
    Ckjh: float = field(init=False)

    def __post_init__(self):
        y = _load_yaml_with_fallback(self.yaml_dir, 'fastener_kv.yaml')
        if y:
            self._PRESETS = _deep_update(dict(self._PRESETS), y)
        self.reset_to_base()

    def reset_to_base(self):
        """核心解析逻辑：定位扣件类型并注入属性"""
        if self.fastener_type not in self._PRESETS:
            raise ValueError(f"在 KV 扣件参数库中未找到类型: '{self.fastener_type}'，请检查输入或扩展字典。")

        conf = self._PRESETS[self.fastener_type]
        for key, value in conf.items():
            setattr(self, key, value)

@dataclass
class FastenerFDKVParams:
    """
    分数阶导数粘弹性扣件 (FDKV) 参数配置
    """
    temperature: int = 20         # 默认环境温度 20℃
    fdkv_switch: str = 'Off'      # FDKV 模块控制开关 ('On' / 'Off')
    yaml_dir: str = 'configs/standard'


    # 嵌套字典：温度 -> FDKV 参数
    _PRESETS: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        -30: {
            'K_alpha': 23.9e6,    # 分数阶刚度系数
            'C_alpha': 12.67e6,   # 分数阶阻尼系数
            'alpha_FDKV': 0.19,   # 分数阶导数阶次
            'IntStep': 160        # 记忆截断积分步数 (Memory Function 步长)
        },
        -10: {
            'K_alpha': 21.41e6, 
            'C_alpha': 8.82e6, 
            'alpha_FDKV': 0.13, 
            'IntStep': 160
        },
        20: {
            'K_alpha': 9.18e6, 
            'C_alpha': 18.79e6, 
            'alpha_FDKV': 0.05, 
            'IntStep': 160
        }
    }, repr=False)

    def __post_init__(self):
        y = _load_yaml_with_fallback(self.yaml_dir, 'fastener_fdkv_params.yaml')
        if y:
            self._PRESETS = _deep_update(dict(self._PRESETS), y)
        self.reset_to_base()

    def reset_to_base(self):
        """核心解析逻辑：根据温度自动提取对应的分数阶本构参数并注入属性"""
        if self.temperature not in self._PRESETS:
            raise ValueError(f"在FDKV扣件参数库中未找到温度: '{self.temperature}℃' 的配置，请检查输入或扩展字典。")

        conf = self._PRESETS[self.temperature]
        for key, value in conf.items():
            setattr(self, key, value)
        self.IntStep = int(self.IntStep)

@dataclass
class RailParams:
    """轨道与钢轨系统基础参数配置"""
    rail_type: str = 'CHN60'  # 默认使用中国 60kg/m 标准钢轨
    yaml_dir: str = 'configs/standard'
    _PRESETS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'CHN60': {
            'mr': 60.64,            # 钢轨每延米质量 (kg/m)
            'E': 2.059e11,          # 钢轨弹性模量 (Pa)
            'Iy': 0.3217e-4,        # 钢轨Y轴惯性矩 (m^4)
            'Iz': 0.524e-5,         # 钢轨Z轴惯性矩 (m^4)
            'I0': 0.3741e-4,        # 钢轨极惯性矩 (m^4)
            'pr': 7860.0,           # 钢轨密度 (kg/m^3)
            'Gr': 2.0e11,           # 钢轨剪切模量 (Pa)
            'vpr': 0.3,             # 钢轨泊松比
            'Gw': 2.0e11,           # 车轮剪切模量 (Pa)
            'vpw': 0.3,             # 车轮泊松比
            'Gk': 0.19587e6,        # 钢轨抗扭转惯性矩
            'aa': 0.082,            # 钢轨支反力作用点到扭转中心的垂直距离 (m)
            'bb': 0.075,            # 钢轨左右侧支反力作用点到扭转中心的水平距离 (m)
            'hr': 0.09453,          # 抗扭中心距轨头距离 (m)
            'gauge': 1435.0,        # 轨距 (mm)
            'rslope': np.arctan(1/40) # 轨底坡 (rad)，直接调用 numpy 计算
        },
        # 未来可继续扩展如 'CHN75' 等重载钢轨
    }, repr=False)

    # 声明所有运行时变量
    mr: float = field(init=False); E: float = field(init=False)
    Iy: float = field(init=False); Iz: float = field(init=False)
    I0: float = field(init=False); pr: float = field(init=False)
    Gr: float = field(init=False); vpr: float = field(init=False)
    Gw: float = field(init=False); vpw: float = field(init=False)
    Gk: float = field(init=False); aa: float = field(init=False)
    bb: float = field(init=False); hr: float = field(init=False)
    gauge: float = field(init=False); rslope: float = field(init=False)

    def __post_init__(self):
        y = _load_yaml_with_fallback(self.yaml_dir, 'rail_params.yaml')
        if y:
            self._PRESETS = _deep_update(dict(self._PRESETS), y)
        self.reset_to_base()
    
    def reset_to_base(self):
        """定位钢轨类型并注入属性"""
        if self.rail_type not in self._PRESETS:
            raise ValueError(f"在参数库中未找到钢轨类型: '{self.rail_type}'")
        conf = self._PRESETS[self.rail_type]
        for key, value in conf.items():
            setattr(self, key, value)

@dataclass
class ExtraForceElements_parameters:
    """额外力元参数配置类，包含非线性力元"""
    Lc: float
    yaml_dir: str = 'configs/standard'
    # ================= Part1.一系垂向减振器 (PVD) =================
    Kz_pvd: float = 875000.0      # 一系垂向减振器刚度
    Cz_pvd: float = 9800.0        # 一系垂向减振器阻尼
    Lpvdx_Bogie: float = 1.491    # 一系垂向减振器至构架中心纵距
    Lpvdx_axlebox: float = 0.241  # 一系垂向减振器至轴箱中心纵距
    dpvd: float = 1.0             # 一系垂向减振器横向跨距之半

    # ================= Part2.转臂结点 (Node) =================
    Knodex: float = 1.71e7        # 转臂结点纵向刚度
    Knodey: float = 6.1e6         # 转臂结点横向刚度
    Knodez: float = 1.71e7        # 转臂结点垂向刚度
    dnode: float = 1.0            # 转臂结点横向跨距之半
    Ht_node: float = 0.04         # 转臂结点上表面距构架质心垂直距离
    Haxlebox_node: float = 0.03   # 转臂结点下表面距轴箱质心垂直距离
    Lnode_axlebox: float = 0.48   # 转臂结点至轴箱中心纵距
    Lnodex_Bogie: float = 0.77    # 转臂结点至构架中心纵距

    # ================= Part3.二系垂向减振器 (SVD) =================
    Kz_svd: float = 0.0           # 二系垂向减振器刚度
    Cz_svd: float = 9800.0        # 二系垂向减振器阻尼
    dsvd: float = 1.323           # 二系垂向减振器横向跨距之半
    Lsvd_Car: float = 7.677       # 二系垂向减振器至车体质心距离
    Lsvd_Bogie: float = 0.55      # 二系垂向减振器至构架质心距离
    Hc_svd: float = 0.94          # 二系垂向减振器车体质心垂向距离

    # ================= Part4.二系横向减振器 (SLD) =================
    Ky_sld: float = 1875000.0     # 二系横向减振器刚度 (3.75e6 / 2)
    Lsld_Bogie: float = 0.15      # SLD纵向上到构架质心的距离
    Ht_sld: float = 0.224         # SLD下表面至构架质心距离
    Hc_sld: float = 0.986         # SLD上表面至车体质心距离
    
    # SLD 阻尼非线性参数点 [速度(m/s), 阻尼力(N)]
    Cy_sld_point1: tuple = (-0.3, -3585.0)
    Cy_sld_point2: tuple = (-0.1, -2500.0)
    Cy_sld_point3: tuple = (0.1, 2500.0)
    Cy_sld_point4: tuple = (0.3, 3585.0)

    # ================= Part5.轴箱参数 =================
    Jaxlebox: float = 2.0         # 轴箱转动惯量

    # ================= 运行时动态生成的数组 =================
    Lsld_Car_L: np.ndarray = field(init=False)    # 左侧SLD纵距数组
    Lsld_Car_R: np.ndarray = field(init=False)    # 右侧SLD纵距数组
    lat_damper_v: np.ndarray = field(init=False)  # 用于插值的速度数组
    lat_damper_f: np.ndarray = field(init=False)  # 用于插值的阻尼力数组

    def __post_init__(self):
        """初始化时结合 Lc 自动计算安装位置矩阵，并提取插值数组"""
        y = _load_yaml_with_fallback(self.yaml_dir, 'extra_force_elements.yaml')
        for k, v in y.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # 兼容 YAML 中点位使用 list 的情况
        for _name in ['Cy_sld_point1', 'Cy_sld_point2', 'Cy_sld_point3', 'Cy_sld_point4']:
            _v = getattr(self, _name)
            if isinstance(_v, list):
                setattr(self, _name, tuple(_v))
        
        # 1. 动态计算 SLD 安装位置 (前后转向架反对称)
        self.Lsld_Car_L = np.array([self.Lc + self.Lsld_Bogie, self.Lc - self.Lsld_Bogie])
        self.Lsld_Car_R = np.array([self.Lc - self.Lsld_Bogie, self.Lc + self.Lsld_Bogie])

        # 2. 拼装非线性阻尼曲线矩阵
        Cy_sld_matrix = np.array([
            self.Cy_sld_point1, 
            self.Cy_sld_point2, 
            self.Cy_sld_point3, 
            self.Cy_sld_point4
        ])
        
        # 3. 切片分离出速度和力，无缝对接 suspension.py 中的 np.interp
        self.lat_damper_v = Cy_sld_matrix[:, 0]
        self.lat_damper_f = Cy_sld_matrix[:, 1]

@dataclass
class Antiyawer_parameters:
    yaml_dir: str = 'configs/standard'
    dsc: float = 1.3   # 抗蛇行减振器横向跨距之半
    kantiyawer: float = 4e6  # 抗蛇行减振器刚度
    point_pre: tuple = (-10.0, -25750.0)
    point1: tuple    = (-0.2, -25750.0)
    point2: tuple    = (-0.03, -20500.0)
    point3: tuple    = (-0.01, -13200.0)
    point4: tuple    = (0.0, 0.0)
    point5: tuple    = (0.01, 13200.0)    
    point6: tuple    = (0.03, 20500.0)   
    point7: tuple    = (0.2, 25750.0)   
    point_end: tuple = (10.0, 25750.0)

    Damper_parameters: np.ndarray = field(init=False)
    yaw_damper_v: np.ndarray = field(init=False)
    yaw_damper_f: np.ndarray = field(init=False)

    def __post_init__(self):
        """初始化时自动拼装矩阵并提取插值所需的一维数组"""
        y = _load_yaml_with_fallback(self.yaml_dir, 'antiyawer_params.yaml')
        for k, v in y.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # 兼容 YAML 中点位使用 list 的情况
        for _name in ['point_pre', 'point1', 'point2', 'point3', 'point4', 'point5', 'point6', 'point7', 'point_end']:
            _v = getattr(self, _name)
            if isinstance(_v, list):
                setattr(self, _name, tuple(_v))

        self.Damper_parameters = np.array([
            self.point_pre, self.point1, self.point2, self.point3,
            self.point4, self.point5, self.point6, self.point7, self.point_end
        ])
        
        self.yaw_damper_v = self.Damper_parameters[:, 0]
        self.yaw_damper_f = self.Damper_parameters[:, 1]

@dataclass
class VehicleParams:
    """
    通用车辆系统参数配置中心 (基于《车辆-轨道耦合动力学》典型参数)
    
    支持的大类 (category):
      - 客车类 (Passenger)
      - 机车类 (Locomotive)
      - 货车类 (Freight)
    支持的车型 (vehicle_type):
      - 客车类: '普通客车', '提速客车', '高速客车'
      - 机车类: '普通机车', '提速机车', '高速机车'
      - 货车类: '普通货车_空车', '普通货车_重车', '25吨轴重货车'
    """
    vehicle_type: str = '高速客车'
    yaml_dir: str = 'configs/standard'
    
    # 预设不同车型的基础参数库 (单位统一为: kg, kg.m^2, N/m, N.s/m, m)
    _PRESETS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        '客车':{
        # ==================== 客车参数 (附表6) ====================
        '普通客车': {
            'Mc_base': 40560.0, 'Mt_base': 3175.0, 'Mw_base': 1900.0,
            'Jcx_base': 1.043e5, 'Jcy_base': 2.966e6, 'Jcz_base': 2.996e6,
            'Jtx_base': 2040.0, 'Jty_base': 2710.0, 'Jtz_base': 4650.0,
            'Jwx_base': 1120.0, 'Jwy_base': 160.0, 'Jwz_base': 1120.0,
            'Kpx_base': 9e6, 'Kpy_base': 4.6e6, 'Kpz_base': 0.805e6,
            'Ksx_base': 0.13e6, 'Ksy_base': 0.09e6, 'Ksz_base': 0.309e6,
            'Cpx_base': 0.0, 'Cpy_base': 0.0, 'Cpz_base': 20e3,
            'Csx_base': 0.0, 'Csy_base': 35e3, 'Csz_base': 40e3,
            'Lc': 9.0, 'Lt': 1.25, 'R': 0.4575, 'Krx': 2.5e6
        },
        '提速客车': {
            'Mc_base': 29600.0, 'Mt_base': 1700.0, 'Mw_base': 1900.0,
            'Jcx_base': 5.802e4, 'Jcy_base': 2.139e6, 'Jcz_base': 2.139e6,
            'Jtx_base': 1600.0, 'Jty_base': 1700.0, 'Jtz_base': 1700.0,
            'Jwx_base': 1067.0, 'Jwy_base': 140.0, 'Jwz_base': 1067.0,
            'Kpx_base': 24e6, 'Kpy_base': 5.1e6, 'Kpz_base': 0.873e6,
            'Ksx_base': 1.2e6, 'Ksy_base': 0.3e6, 'Ksz_base': 0.41e6,
            'Cpx_base': 0.0, 'Cpy_base': 0.0, 'Cpz_base': 30e3,
            'Csx_base': 0.0, 'Csy_base': 25e3, 'Csz_base': 108.7e3,
            'Lc': 9.0, 'Lt': 1.2, 'R': 0.4575, 'Krx': 2.5e6
        },
        '高速客车': {
            'Mc_base': 34000.0, 'Mt_base': 3000.0, 'Mw_base': 1400.0,
            'Jcx_base': 7.506e4, 'Jcy_base': 2.277e6, 'Jcz_base': 2.086e6,
            'Jtx_base': 2260.0, 'Jty_base': 2710.0, 'Jtz_base': 3160.0,
            'Jwx_base': 915.0, 'Jwy_base': 140.0, 'Jwz_base': 915.0,
            'Kpx_base': 10e6, 'Kpy_base': 5e6, 'Kpz_base': 0.55e6,
            'Ksx_base': 0.15e6, 'Ksy_base': 0.15e6, 'Ksz_base': 0.4e6,
            'Cpx_base': 0.0, 'Cpy_base': 0.0, 'Cpz_base': 6e3,
            'Csx_base': 0.0, 'Csy_base': 60e3, 'Csz_base': 80e3,
            'Lc': 9.0, 'Lt': 1.2, 'R': 0.4575, 'Krx': 2.5e6
        }},
        '机车': {
        # ==================== 机车参数 (附表8) ====================
        '普通机车': {
            'Mc_base': 72452.0, 'Mt_base': 15293.0, 'Mw_base': 5827.0,
            'Jcx_base': 1.604e5, 'Jcy_base': 1.71e6, 'Jcz_base': 1.92e6,
            'Jtx_base': 6263.7, 'Jty_base': 59091.0, 'Jtz_base': 39699.0,
            'Jwx_base': 2913.0, 'Jwy_base': 408.0, 'Jwz_base': 4126.0,
            'Kpx_base': 54.55e6, 'Kpy_base': 4.74e6, 'Kpz_base': 9.615e6,
            'Ksx_base': 0.522e6, 'Ksy_base': 0.46e6, 'Ksz_base': 2.679e6,
            'Cpx_base': 0.0, 'Cpy_base': 0.0, 'Cpz_base': 100e3,
            'Csx_base': 0.0, 'Csy_base': 60e3, 'Csz_base': 80e3,
            'Lc': 5.642, 'Lt': 2.15, 'R': 0.625
        },
        '提速机车': {
            'Mc_base': 63400.0, 'Mt_base': 20563.0, 'Mw_base': 3239.0,
            'Jcx_base': 1.435e5, 'Jcy_base': 1.521e6, 'Jcz_base': 1.718e6,
            'Jtx_base': 7370.0, 'Jty_base': 73274.0, 'Jtz_base': 78243.0,
            'Jwx_base': 2450.0, 'Jwy_base': 405.0, 'Jwz_base': 2450.0,
            'Kpx_base': 20.0e6, 'Kpy_base': 5.5e6, 'Kpz_base': 2.15e6,
            'Ksx_base': 0.426e6, 'Ksy_base': 0.426e6, 'Ksz_base': 1.596e6,
            'Cpx_base': 0.0, 'Cpy_base': 0.0, 'Cpz_base': 80e3,
            'Csx_base': 0.0, 'Csy_base': 90e3, 'Csz_base': 120e3,
            'Lc': 5.39, 'Lt': 2.15, 'R': 0.625
        },
        '高速机车': {
            'Mc_base': 59364.2, 'Mt_base': 5630.8, 'Mw_base': 1843.5,
            'Jcx_base': 1.305e5, 'Jcy_base': 1.723e6, 'Jcz_base': 1.796e6,
            'Jtx_base': 2202.0, 'Jty_base': 9487.0, 'Jtz_base': 11233.0,
            'Jwx_base': 1263.0, 'Jwy_base': 219.0, 'Jwz_base': 1285.0,
            'Kpx_base': 30.8e6, 'Kpy_base': 4.878e6, 'Kpz_base': 2.3996e6,
            'Ksx_base': 0.3156e6, 'Ksy_base': 0.3156e6, 'Ksz_base': 0.8858e6,
            'Cpx_base': 0.0, 'Cpy_base': 0.0, 'Cpz_base': 30e3,
            'Csx_base': 0.0, 'Csy_base': 50e3, 'Csz_base': 45e3,
            'Lc': 5.73, 'Lt': 1.5, 'R': 0.525
        }},
        '货车': {
        # ==================== 货车参数 (附表7) ====================
        # 注：三大件转向架使用 M1(侧架), MB(摇枕)。为兼容常规代码结构，Mt_base 取 M1 质量，
        # 实际动力学方程组装时需根据货车拓扑结构使用 MB 和 M1。
        '普通货车_重车': {
            'Mc_base': 77000.0, 'Mt_base': 330.0, 'MB_base': 470.0, 'Mw_base': 1200.0,
            'Jcx_base': 1e5, 'Jcy_base': 1.2e6, 'Jcz_base': 1.07e6,
            'J1y_base': 100.0, 'J1z_base': 80.0, 'JBz_base': 190.0, # 侧架与摇枕惯量
            'Jtx_base': 0.0, 'Jty_base': 0.0, 'Jtz_base': 0.0,      # 三大件无整体构架惯量
            'Jwx_base': 740.0, 'Jwy_base': 100.0, 'Jwz_base': 740.0,
            'Kpx_base': 0.0, 'Kpy_base': 0.0, 'Kpz_base': 0.0,      # 表格中为空
            'Ksx_base': 4.14e6, 'Ksy_base': 4.14e6, 'Ksz_base': 5.32e6,
            'Ksz1_base': 0.769e6,                                   # 楔块弹簧垂向刚度
            'Cpx_base': 0.0, 'Cpy_base': 0.0, 'Cpz_base': 0.0,
            'Csx_base': 0.0, 'Csy_base': 0.0, 'Csz_base': 0.0,
            'Lc': 4.25, 'Lt': 0.875, 'R': 0.42
        },
        '普通货车_空车': {
            'Mc_base': 14600.0, 'Mt_base': 330.0, 'MB_base': 470.0, 'Mw_base': 1200.0,
            'Jcx_base': 2.66e4, 'Jcy_base': 2.66e5, 'Jcz_base': 2.84e5,
            'J1y_base': 100.0, 'J1z_base': 80.0, 'JBz_base': 190.0,
            'Jtx_base': 0.0, 'Jty_base': 0.0, 'Jtz_base': 0.0,
            'Jwx_base': 740.0, 'Jwy_base': 100.0, 'Jwz_base': 740.0,
            'Kpx_base': 0.0, 'Kpy_base': 0.0, 'Kpz_base': 0.0,
            'Ksx_base': 4.14e6, 'Ksy_base': 4.14e6, 'Ksz_base': 5.32e6,
            'Ksz1_base': 0.769e6,
            'Cpx_base': 0.0, 'Cpy_base': 0.0, 'Cpz_base': 0.0,
            'Csx_base': 0.0, 'Csy_base': 0.0, 'Csz_base': 0.0,
            'Lc': 4.25, 'Lt': 0.875, 'R': 0.42
        },
        '25吨轴重货车': {
            'Mc_base': 90000.0, 'Mt_base': 460.0, 'MB_base': 596.0, 'Mw_base': 1145.0,
            'Jcx_base': 1.148e5, 'Jcy_base': 1.381e6, 'Jcz_base': 1.4075e6,
            'J1y_base': 192.0, 'J1z_base': 175.0, 'JBz_base': 244.0,
            'Jtx_base': 0.0, 'Jty_base': 0.0, 'Jtz_base': 0.0,
            'Jwx_base': 700.0, 'Jwy_base': 100.0, 'Jwz_base': 700.0,
            'Kpx_base': 3.6e6, 'Kpy_base': 3.6e6, 'Kpz_base': 17e6,
            'Ksx_base': 6.5e6, 'Ksy_base': 6.5e6, 'Ksz_base': 5.9e6,
            'Ksz1_base': 0.769e6,
            'Cpx_base': 0.0, 'Cpy_base': 0.0, 'Cpz_base': 3e3,
            'Csx_base': 0.0, 'Csy_base': 0.0, 'Csz_base': 3e3,
            'Lc': 4.6, 'Lt': 0.915, 'R': 0.42
        }}
    }, repr=False)

    category: str = field(init=False) # 车辆大类 (Passenger, Locomotive, Freight)

    # ================= 运行时动态生成的变量 =================
    Mc: float = field(init=False); Mt: float = field(init=False); Mw: float = field(init=False)
    MB: float = field(init=False) # 货车摇枕质量
    
    Jcx: float = field(init=False); Jcy: float = field(init=False); Jcz: float = field(init=False)
    Jtx: float = field(init=False); Jty: float = field(init=False); Jtz: float = field(init=False)
    Jwx: float = field(init=False); Jwy: float = field(init=False); Jwz: float = field(init=False)
    J1y: float = field(init=False); J1z: float = field(init=False); JBz: float = field(init=False) # 货车侧架/摇枕惯量
    
    Kpx: float = field(init=False); Kpy: float = field(init=False); Kpz: float = field(init=False)
    Cpx: float = field(init=False); Cpy: float = field(init=False); Cpz: float = field(init=False)
    Ksx: float = field(init=False); Ksy: float = field(init=False); Ksz: float = field(init=False)
    Csx: float = field(init=False); Csy: float = field(init=False); Csz: float = field(init=False)
    Ksz1: float = field(init=False) # 货车楔块弹簧垂向刚度
    
    # 几何常量与衍生常量
    HcB: float = 1.415; HBt: float = -0.081; Htw: float = 0.14
    dw: float = 0.978; ds: float = 0.978
    Lc: float = field(init=False); Lt: float = field(init=False)
    R: float = field(init=False); Krx: float = field(init=False)

    G: float = field(init=False)  # 接触常数
    P0: float = field(init=False) # 静轮重

    def __post_init__(self):
        """初始化加载对应车型"""
        y = _load_yaml_with_fallback(self.yaml_dir, 'vehicle_params.yaml')
        if y:
            self._PRESETS = _deep_update(dict(self._PRESETS), y)
        self.reset_to_base()

    def reset_to_base(self):
        found_category = False
        conf = None

        for catname, models in self._PRESETS.items():
            if self.vehicle_type in models:
                self.category = catname
                found_category = True
                conf = models[self.vehicle_type]
                break
            
        if not conf:
            raise ValueError(f"在参数库中未找到车型: '{self.vehicle_type}'，请检查拼写或扩展参数库。")
        
        # self.category = found_category

        # 批量注入基础属性
        for key, value in conf.items():
            setattr(self, key, value)
            
        # 核心质量/惯量/刚度映射到运行变量
        self.Mc, self.Mt, self.Mw = getattr(self, 'Mc_base', 0), getattr(self, 'Mt_base', 0), getattr(self, 'Mw_base', 0)
        self.MB = getattr(self, 'MB_base', 0)
        
        self.Jcx, self.Jcy, self.Jcz = getattr(self, 'Jcx_base', 0), getattr(self, 'Jcy_base', 0), getattr(self, 'Jcz_base', 0)
        self.Jtx, self.Jty, self.Jtz = getattr(self, 'Jtx_base', 0), getattr(self, 'Jty_base', 0), getattr(self, 'Jtz_base', 0)
        self.Jwx, self.Jwy, self.Jwz = getattr(self, 'Jwx_base', 0), getattr(self, 'Jwy_base', 0), getattr(self, 'Jwz_base', 0)
        self.J1y, self.J1z, self.JBz = getattr(self, 'J1y_base', 0), getattr(self, 'J1z_base', 0), getattr(self, 'JBz_base', 0)
        
        self.Kpx, self.Kpy, self.Kpz = getattr(self, 'Kpx_base', 0), getattr(self, 'Kpy_base', 0), getattr(self, 'Kpz_base', 0)
        self.Ksx, self.Ksy, self.Ksz = getattr(self, 'Ksx_base', 0), getattr(self, 'Ksy_base', 0), getattr(self, 'Ksz_base', 0)
        self.Ksz1 = getattr(self, 'Ksz1_base', 0)
        
        self.Cpx, self.Cpy, self.Cpz = getattr(self, 'Cpx_base', 0), getattr(self, 'Cpy_base', 0), getattr(self, 'Cpz_base', 0)
        self.Csx, self.Csy, self.Csz = getattr(self, 'Csx_base', 0), getattr(self, 'Csy_base', 0), getattr(self, 'Csz_base', 0)

        self._calculate_derived_params()

    def randomize_for_dataset(self, noise_ratio: float = 0.15):
        """比例加噪，并刷新衍生参数"""
        def get_noise(base_val):
            return base_val * (1 + (np.random.rand() * 2 - 1) * noise_ratio)

        for attr in ['Mc', 'Mt', 'Mw', 'MB', 'Jcx', 'Jcy', 'Jcz', 'Jtx', 'Jty', 'Jtz', 'Jwx', 'Jwy', 'Jwz', 
                     'Kpx', 'Kpy', 'Kpz', 'Ksx', 'Ksy', 'Ksz', 'Cpz', 'Csy', 'Csz']:
            if hasattr(self, attr) and getattr(self, f"{attr}_base", 0) != 0:
                setattr(self, attr, get_noise(getattr(self, f"{attr}_base")))

        self._calculate_derived_params()

    def _calculate_derived_params(self):
        """
        计算静轮重 P0 时需区分客机车 (2构架) 与货车 (4侧架+2摇枕)
        """
        self.G = 3.86 * (self.R ** -0.115) * 1e-8 
        if self.category == '货车':
            # 货车总重 = 车体 + 2*摇枕 + 4*侧架 + 4*轮对
            self.P0 = (self.Mc + 2 * self.MB + 4 * self.Mt + 4 * self.Mw) * 9.81 / 8
        else:
            # 客/机车总重 = 车体 + 2*构架 + 4*轮对
            self.P0 = (self.Mc + 2 * self.Mt + 4 * self.Mw) * 9.81 / 8