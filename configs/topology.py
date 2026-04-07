import numpy as np
from dataclasses import dataclass, field

@dataclass
class StepState:
    """当前积分步的物理状态集合 (解耦后的 X 和 V)"""
    # 1. 车辆总状态
    XCar: np.ndarray; VCar: np.ndarray
    # 2. 轮对独立状态 (长度为 4 的 1D 数组)
    X_YW: np.ndarray; X_ZW: np.ndarray; X_RollW: np.ndarray; X_SpinW: np.ndarray; X_YawW: np.ndarray
    V_XW: np.ndarray; V_YW: np.ndarray; V_ZW: np.ndarray; V_RollW: np.ndarray; V_SpinW: np.ndarray; V_YawW: np.ndarray
    # 3. 钢轨总状态
    XRail_L: np.ndarray; VRail_L: np.ndarray
    XRail_R: np.ndarray; VRail_R: np.ndarray
    # 4. 钢轨分层状态 (沉浮 Z, 横移 Y, 扭转 T)
    XRail_Z_L: np.ndarray; XRail_Y_L: np.ndarray; XRail_T_L: np.ndarray
    VRail_Z_L: np.ndarray; VRail_Y_L: np.ndarray; VRail_T_L: np.ndarray
    XRail_Z_R: np.ndarray; XRail_Y_R: np.ndarray; XRail_T_R: np.ndarray
    VRail_Z_R: np.ndarray; VRail_Y_R: np.ndarray; VRail_T_R: np.ndarray
    # 5. 轨枕状态 (沉浮 Z, 横移 Y, 侧滚 Roll)
    XSleeper_Z: np.ndarray; XSleeper_Y: np.ndarray; XSleeper_Roll: np.ndarray
    VSleeper_Z: np.ndarray; VSleeper_Y: np.ndarray; VSleeper_Roll: np.ndarray
    # 6. 路基状态
    XSubgrade_L: np.ndarray; VSubgrade_L: np.ndarray
    XSubgrade_R: np.ndarray; VSubgrade_R: np.ndarray

@dataclass
class SystemTopology:
    """
    /系统拓扑管理模块/ 
    计算并管理耦合系统各个子结构(车体、钢轨、轨下)的自由度(DOF)数量与索引映射
    同时负责前向预分配积分所需的核心矩阵
    """
    # 外部依赖参量 (需传入)
    Nt: int                     # 积分总步数
    Nsub: int                   # 柔性轨道子结构节点数 (如扣件数)
    NV: int = 1                 # 钢轨模态控制数 - 沉浮
    NL: int = 1                 # 钢轨模态控制数 - 横移
    NT: int = 1                 # 钢轨模态控制数 - 扭转

    # 以下为类内部自动推导的拓扑参量
    Fnum_Total: int = field(init=False)
    
    # 子结构切片索引 (Python 切片左闭右开，因此存成 (start, end) 元组)
    idx_Car: tuple = field(init=False)
    idx_Rail_L: tuple = field(init=False)
    idx_Rail_R: tuple = field(init=False)
    idx_Sleeper: tuple = field(init=False)
    idx_Subgrade_L: tuple = field(init=False)
    idx_Subgrade_R: tuple = field(init=False)

    def __post_init__(self):
        """计算耦合系统各个部分的自由度数量与 Python 切片索引 (0-based)"""
        # 1. 计算各部分自由度数量
        Fnum_Car = 35                                    # 车体自由度(不含含轴箱) = 35
        Fnum_Rail_L = self.NV + self.NL + self.NT           # 左钢轨模态自由度
        Fnum_Rail_R = self.NV + self.NL + self.NT           # 右钢轨模态自由度
        Fnum_Sleeper = 3 * (self.Nsub + 1)                  # 轨枕自由度(沉浮，横移，转动)
        Fnum_Subgrade_L = self.Nsub + 1                     # 左路基自由度
        Fnum_Subgrade_R = self.Nsub + 1                     # 右路基自由度
        
        self.Fnum_Total = (Fnum_Car + Fnum_Rail_L + Fnum_Rail_R + 
                           Fnum_Sleeper + Fnum_Subgrade_L + Fnum_Subgrade_R)

        # 2. 生成切片索引 (0 开始，左闭右开)
        current_idx = 0
        
        self.idx_Car = (current_idx, current_idx + Fnum_Car)
        current_idx += Fnum_Car
        
        self.idx_Rail_L = (current_idx, current_idx + Fnum_Rail_L)
        current_idx += Fnum_Rail_L
        
        self.idx_Rail_R = (current_idx, current_idx + Fnum_Rail_R)
        current_idx += Fnum_Rail_R
        
        self.idx_Sleeper = (current_idx, current_idx + Fnum_Sleeper)
        current_idx += Fnum_Sleeper
        
        self.idx_Subgrade_L = (current_idx, current_idx + Fnum_Subgrade_L)
        current_idx += Fnum_Subgrade_L
        
        self.idx_Subgrade_R = (current_idx, current_idx + Fnum_Subgrade_R)
        
    def allocate_memory(self, switch_2point_contact: str = 'On'):
        """
        预分配主程序和监视模块所需的大型矩阵矩阵。
        返回一个包含各种数组引用的字典。
        """
        Nt = self.Nt
        
        # 1. 核心状态量矩阵 (位移 X, 速度 V, 加速度 A)
        X = np.zeros((Nt, self.Fnum_Total))
        V = np.zeros((Nt, self.Fnum_Total))
        A = np.zeros((Nt, self.Fnum_Total))
        
        # 2. 扣件压缩历史 (FDKV 模型所需)
        # 长度为 Nsub + 1 的扣件节点列
        pad_len = self.Nsub + 1
        PadComp_L1 = np.zeros((Nt, pad_len))
        PadComp_L2 = np.zeros((Nt, pad_len))
        PadComp_R1 = np.zeros((Nt, pad_len))
        PadComp_R2 = np.zeros((Nt, pad_len))

        # 3. 监视模块变量
        spy_dict = {
            'Yixi_Force_x': np.zeros((Nt, 8)),
            'Yixi_Force_y': np.zeros((Nt, 8)),
            'Yixi_Force_z': np.zeros((Nt, 8)),
            'Erxi_Force_x': np.zeros((Nt, 4)),
            'Erxi_Force_y': np.zeros((Nt, 4)),
            'Erxi_Force_z': np.zeros((Nt, 4)),
            'FV_Fastener': np.zeros((Nt, 6)),
            'FL_Fastener': np.zeros((Nt, 2)),
            # 根据两点接触开关动态分配轮轨力监视器
            'TotalVerticalForce': np.zeros((Nt, 8)),
            'TotalLateralForce': np.zeros((Nt, 8)),
            'ContactInfo_On_Wheel': np.zeros((Nt, 8 if switch_2point_contact == 'Off' else 16)),
            'ContactInfo_On_Rail': np.zeros((Nt, 8 if switch_2point_contact == 'Off' else 16))
        }
        
        if switch_2point_contact == 'Off':
            spy_dict['RawCreepForce'] = np.zeros((Nt, 24))
            spy_dict['RawKsi'] = np.zeros((Nt, 24))
        else:
            spy_dict['TotalVerticalForce_Point2'] = np.zeros((Nt, 8))
            spy_dict['TotalLateralForce_Point2'] = np.zeros((Nt, 8))
            spy_dict['RawCreepForce_Store'] = np.zeros((Nt, 24))
            spy_dict['RawCreepForce_Point2_Store'] = np.zeros((Nt, 24))
            spy_dict['RawKsi_Store'] = np.zeros((Nt, 24))
            spy_dict['RawKsi_Point2_Store'] = np.zeros((Nt, 24))

        return X, V, A, PadComp_L1, PadComp_L2, PadComp_R1, PadComp_R2, spy_dict
    def extract_state(self, X_moment: np.ndarray, V_moment: np.ndarray, Vc: float) -> StepState:
        """
        极速状态提取器：将一维的总状态向量 X, V 拆解为各子结构的状态数组。
        利用 NumPy 切片步长特性，消灭 for 循环，性能最大化。
        """
        # ==========================================
        # 1. 车辆系统提取
        # ==========================================
        XCar = X_moment[self.idx_Car[0] : self.idx_Car[1]]
        VCar = V_moment[self.idx_Car[0] : self.idx_Car[1]]
        
        # ==========================================
        # 2. 轮对状态提取 (高光时刻：用步长切片替代 MATLAB 的 for 循环)
        # ==========================================
        # 轮对数据在 XCar 中的索引(0-based): 15~34
        # XCar[15:35:5] -> 提取 15, 20, 25, 30 (四个轮对的横向位移 YW)
        X_YW = XCar[15:35:5]; V_YW = VCar[15:35:5]
        X_ZW = XCar[16:35:5]; V_ZW = VCar[16:35:5]
        X_RollW = XCar[17:35:5]; V_RollW = VCar[17:35:5]
        X_SpinW = XCar[18:35:5]; V_SpinW = VCar[18:35:5]
        X_YawW = XCar[19:35:5]; V_YawW = VCar[19:35:5]
        
        # 车辆前进速度 V_XW 恒等于车速 Vc
        V_XW = np.full(4, Vc)
        # ==========================================
        # 3. 钢轨部分提取
        # ==========================================
        XRail_L = X_moment[self.idx_Rail_L[0] : self.idx_Rail_L[1]]
        VRail_L = V_moment[self.idx_Rail_L[0] : self.idx_Rail_L[1]]
        XRail_R = X_moment[self.idx_Rail_R[0] : self.idx_Rail_R[1]]
        VRail_R = V_moment[self.idx_Rail_R[0] : self.idx_Rail_R[1]]
        
        # 钢轨内部自由度拆分 (NV: 沉浮, NL: 横移, NT: 扭转)
        idx_z, idx_y, idx_t = self.NV, self.NV + self.NL, self.NV + self.NL + self.NT
        XRail_Z_L, XRail_Y_L, XRail_T_L = np.split(XRail_L, [idx_z, idx_y])
        VRail_Z_L, VRail_Y_L, VRail_T_L = np.split(VRail_L, [idx_z, idx_y])
        XRail_Z_R, XRail_Y_R, XRail_T_R = np.split(XRail_R, [idx_z, idx_y])
        VRail_Z_R, VRail_Y_R, VRail_T_R = np.split(VRail_R, [idx_z, idx_y])

        # ==========================================
        # 4. 轨下结构部分提取 (轨枕 + 路基)
        # ==========================================
        XSleeper = X_moment[self.idx_Sleeper[0] : self.idx_Sleeper[1]]
        VSleeper = V_moment[self.idx_Sleeper[0] : self.idx_Sleeper[1]]
        
        ns = self.Nsub + 1
        # np.split 按照节点将数组均分为 3 段 (Z, Y, Roll)
        XSleeper_Z, XSleeper_Y, XSleeper_Roll = np.split(XSleeper, 3)
        VSleeper_Z, VSleeper_Y, VSleeper_Roll = np.split(VSleeper, 3)
        
        XSubgrade_L = X_moment[self.idx_Subgrade_L[0] : self.idx_Subgrade_L[1]]
        VSubgrade_L = V_moment[self.idx_Subgrade_L[0] : self.idx_Subgrade_L[1]]
        
        XSubgrade_R = X_moment[self.idx_Subgrade_R[0] : self.idx_Subgrade_R[1]]
        VSubgrade_R = V_moment[self.idx_Subgrade_R[0] : self.idx_Subgrade_R[1]]

        # ==========================================
        # 5. 打包返回
        # ==========================================
        return StepState(
            XCar, VCar, 
            X_YW, X_ZW, X_RollW, X_SpinW, X_YawW,
            V_XW, V_YW, V_ZW, V_RollW, V_SpinW, V_YawW,
            XRail_L, VRail_L, XRail_R, VRail_R,
            XRail_Z_L, XRail_Y_L, XRail_T_L, VRail_Z_L, VRail_Y_L, VRail_T_L,
            XRail_Z_R, XRail_Y_R, XRail_T_R, VRail_Z_R, VRail_Y_R, VRail_T_R,
            XSleeper_Z, XSleeper_Y, XSleeper_Roll, VSleeper_Z, VSleeper_Y, VSleeper_Roll,
            XSubgrade_L, VSubgrade_L, XSubgrade_R, VSubgrade_R
        )