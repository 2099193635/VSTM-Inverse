'''
Author: Niscienc 60505912+2099193635@users.noreply.github.com
Date: 2026-03-04 16:35:05
LastEditors: Niscienc 60505912+2099193635@users.noreply.github.com
LastEditTime: 2026-03-14 20:00:01
FilePath: \VTCM_PYTHON\defect_injector\irregularity.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np
from power_spectrum.china import PowerSpectrum_ChineseHighSpeed, PSD_China_GanXian
from power_spectrum.German import PowerSpectrum_German
from power_spectrum.American import PSD_America
from defect_injector.local_defects import weld_bead
import pandas as pd


class Irregularity:
    """
    type: 不平顺类型 (如 '随机不平顺')
    LoadType: 载荷类型 (施加横向还是垂向)
    Type2: 载荷谱类型 (是时间谱还是空间谱)
    Tstart: 载荷开始时间 (s)
    Av_harmonic: 垂向谐波不平顺幅值 (m)
    Lv_harmonic: 垂向谐波不平顺波长 (m)
    Al_harmonic: 横向谐波不平顺幅值 (m)
    Ll_harmonic: 横向谐波不平顺波长 (m)
    Factor: 单位转换因子(输入不平顺样本单位为mm)
    """
    def __init__(self,
                 Lc: float,
                 Lt: float,
                 Vc: float,
                 Tstep: float,
                 Tz: float,
                 Nt: int,
                 type: str = '随机不平顺',
                 LoadType: int = 3,
                 Type2: str = '时间谱',
                 Tstart: float = 1,
                 Av_harmonic: float = 0.008,
                 Lv_harmonic: float = 10,
                 Al_harmonic=0.006,
                 Ll_harmonic=10,
                 Factor = 0.001,
                 mile: float = 2000,
                 lanm_min: float = 1.5,
                 lanm_max: float = 120,
                 powerSpectrum_type = '高铁谱',
                 input_path: str = '',
                 output_path: str = '',
                 external_mileage_mode: str = 'relative',
                 external_origin_abs: float = None,
                 external_distance_unit: str = 'auto',
                 external_start_mileage: float = None
                 ):
        self.Lc = Lc
        self.Lt = Lt
        self.Vc = Vc
        self.Tstep = Tstep
        self.Tz = Tz
        self.Nt = Nt
        self.type = type
        self.LoadType = LoadType
        self.Type2 = Type2
        self.Tstart = Tstart
        self.Av_harmonic = Av_harmonic
        self.Lv_harmonic = Lv_harmonic
        self.Al_harmonic = Al_harmonic
        self.Ll_harmonic = Ll_harmonic
        self.Factor = Factor
        self.mile = mile
        self.lanm_min = lanm_min
        self.lanm_max = lanm_max
        self.powerSpectrum_type = powerSpectrum_type
        self.input_path = input_path
        self.output_path = output_path
        # 外部导入里程解释模式：
        #  - 'relative': 文件第1列视为相对里程（推荐，通常从0开始）
        #  - 'absolute': 文件第1列视为绝对里程，需要结合 external_origin_abs 换算
        self.external_mileage_mode = external_mileage_mode
        self.external_origin_abs = external_origin_abs
        # 外部里程单位：'m' / 'km' / 'auto'
        self.external_distance_unit = external_distance_unit
        # 外部导入的起始里程（可选）
        # - 为 None 时：按 origin_abs(absolute) 或文件首值(relative)作为起点
        # - 非 None 时：以该里程作为 t=0 起点（同样支持 external_files 覆盖）
        self.external_start_mileage = external_start_mileage

    def prepare_external_irre_data(self):
        df = pd.read_csv(self.input_path)
        relative_distance = df['Mileage'].values 

    def _external_distance_to_time(self, distance_arr: np.ndarray, external_files: dict = None) -> np.ndarray:
        """
        将外部导入文件第1列里程统一转换到仿真时间轴。

        规则：
        1) relative 模式：默认以文件首值为0基准（防止用户文件非零起点）；
        2) absolute 模式：优先使用 external_files['origin_abs'] 或 self.external_origin_abs 作为绝对起点；
           若未提供则退化为文件首值。
        """
        s = np.asarray(distance_arr, dtype=float)
        if s.size == 0:
            return s

        s_raw = s.copy()

        # 单位归一：统一换算到 m
        unit = str(self.external_distance_unit).lower()
        use_km_scale = False
        if unit == 'km':
            use_km_scale = True
        elif unit == 'auto':
            # 启发式判断：若绝对值量级像“公里标”(如 271.8~278.3)，按 km 处理
            # 若已是米级绝对里程(如 271000~278000)，则不转换。
            if np.nanmax(np.abs(s_raw)) < 1.0e4:
                use_km_scale = True

        if use_km_scale:
            s = s * 1000.0

        def _convert_single_distance_value(v):
            if v is None:
                return None
            vv = float(v)
            if use_km_scale:
                return vv * 1000.0
            return vv

        mode = str(self.external_mileage_mode).lower()
        ext = external_files or {}
        start_mileage = ext.get('start_mileage', self.external_start_mileage)
        start_mileage = _convert_single_distance_value(start_mileage)

        if start_mileage is not None:
            # 用户明确指定起始里程，优先级最高
            s_rel = s - float(start_mileage)
        elif mode == 'absolute':
            origin_abs = ext.get('origin_abs', self.external_origin_abs)
            origin_abs = _convert_single_distance_value(origin_abs)
            if origin_abs is None:
                origin_abs = float(s[0])
            s_rel = s - float(origin_abs)
        else:
            # relative 模式也强制平移到0，避免用户给出“相对但不从0开始”的文件导致时间错位
            s_rel = s - float(s[0])

        return s_rel / self.Vc

    def excitation_irregularity(self, defect_switch: str = 'on', external_files: dict = None):
        """
        激励模块 - 生成多轮对时空耦合的不平顺及变化率矩阵
        defect_switch: 是否开启局部病害 ('on'/'off')
        external_files: 当 type='外部导入' 时必填，传入包含4个文件路径的字典
        返回: 
            bz_L, by_L, dbz_L, dby_L: 左轨 垂/横向位移及速度 (4 x (Nt+1) 矩阵)
            bz_R, by_R, dbz_R, dby_R: 右轨 垂/横向位移及速度 (4 x (Nt+1) 矩阵)
            a, L: 局部病害参数
        """
        # ==========================================
        # 1. 基础控制参量与轮对时延步数计算
        # ==========================================
        FlowDimension = int(np.round(1.5 * self.Nt))
        TimeSequence = np.arange(1, FlowDimension + 1) * self.Tstep
        a, L = 0, 0
        
        # 轮对 1, 2, 3 相对于轮对 4 的延迟时间步数
        step1 = int(np.round(2 * (self.Lc + self.Lt) / self.Vc / self.Tstep))
        step2 = int(np.round(2 * self.Lc / self.Vc / self.Tstep))
        step3 = int(np.round(2 * self.Lt / self.Vc / self.Tstep))

        # ==========================================
        # 2. 激扰源生成 (分为 解析谐波 vs 数值随机谱)
        # ==========================================
        if self.type == '谐波不平顺':
            # --- 谐波分支：使用高精度解析导数，无需平滑和差分 ---
            from defect_injector.local_defects import excitation_harmonic
            
            # 这里的 self.Tstart 对应 MATLAB 中的 Lstart (空白启动时间)
            bz, dbz = excitation_harmonic(self.Av_harmonic, self.Lv_harmonic, self.Vc, self.Tstep, self.Tz, self.Tstart)
            by, dby = excitation_harmonic(self.Al_harmonic, self.Ll_harmonic, self.Vc, self.Tstep, self.Tz, self.Tstart)
            
            # 原版 MATLAB 设定横向激扰取反
            by, dby = -by, -dby
            
            # 修复 MATLAB 原版 Bug：未向左右轨赋值
            bpsz_L, bpsz_R = bz.copy(), bz.copy()
            dbpsz_L, dbpsz_R = dbz.copy(), dbz.copy()
            bpsy_L, bpsy_R = by.copy(), by.copy()
            dbpsy_L, dbpsy_R = dby.copy(), dby.copy()

        else:
            # --- 数值分支：包含随机谱、外部导入、无不平顺 ---
            if self.type == '随机不平顺':
                GD_L, GD_R, GX_L, GX_R, a, L = self.irregularity_pad_generation(defect_switch)
                limit = len(TimeSequence)
                VL_Spline = GD_L[:limit, 1] * self.Factor
                VR_Spline = GD_R[:limit, 1] * self.Factor
                LL_Spline = GX_L[:limit, 1] * self.Factor
                LR_Spline = GX_R[:limit, 1] * self.Factor

            elif self.type == '外部导入':
                from scipy.interpolate import PchipInterpolator
                if not external_files:
                    raise ValueError(" -> [错误] 外部导入时必须传入 external_files 字典！")
                
                data_VL = np.loadtxt(external_files['VL'])
                data_VR = np.loadtxt(external_files['VR'])
                data_LL = np.loadtxt(external_files['LL'])
                data_LR = np.loadtxt(external_files['LR'])
                
                if self.Type2 == '空间谱':
                    time_VL = self._external_distance_to_time(data_VL[:, 0], external_files)
                    time_VR = self._external_distance_to_time(data_VR[:, 0], external_files)
                    time_LL = self._external_distance_to_time(data_LL[:, 0], external_files)
                    time_LR = self._external_distance_to_time(data_LR[:, 0], external_files)

                    # 关键修复：
                    # 1) 禁止超出外部数据范围外推；
                    # 2) 使用形状保持插值(PCHIP)减少三次样条的过冲；
                    # 3) 再次限幅到原始数据范围，避免生成“原数据不存在的极值”。
                    p_vl = PchipInterpolator(time_VL, data_VL[:, 1], extrapolate=False)
                    p_vr = PchipInterpolator(time_VR, data_VR[:, 1], extrapolate=False)
                    p_ll = PchipInterpolator(time_LL, data_LL[:, 1], extrapolate=False)
                    p_lr = PchipInterpolator(time_LR, data_LR[:, 1], extrapolate=False)

                    VL_Spline = np.nan_to_num(p_vl(TimeSequence), nan=0.0)
                    VR_Spline = np.nan_to_num(p_vr(TimeSequence), nan=0.0)
                    LL_Spline = np.nan_to_num(p_ll(TimeSequence), nan=0.0)
                    LR_Spline = np.nan_to_num(p_lr(TimeSequence), nan=0.0)

                    VL_Spline = np.clip(VL_Spline, np.nanmin(data_VL[:, 1]), np.nanmax(data_VL[:, 1])) * self.Factor
                    VR_Spline = np.clip(VR_Spline, np.nanmin(data_VR[:, 1]), np.nanmax(data_VR[:, 1])) * self.Factor
                    LL_Spline = np.clip(LL_Spline, np.nanmin(data_LL[:, 1]), np.nanmax(data_LL[:, 1])) * self.Factor
                    LR_Spline = np.clip(LR_Spline, np.nanmin(data_LR[:, 1]), np.nanmax(data_LR[:, 1])) * self.Factor
                elif self.Type2 in ('时间谱', '时间序列'):
                    limit = min(len(TimeSequence), len(data_VL))
                    VL_Spline, VR_Spline, LL_Spline, LR_Spline = [np.zeros(len(TimeSequence)) for _ in range(4)]
                    VL_Spline[:limit] = data_VL[:limit, 1] * self.Factor
                    VR_Spline[:limit] = data_VR[:limit, 1] * self.Factor
                    LL_Spline[:limit] = data_LL[:limit, 1] * self.Factor
                    LR_Spline[:limit] = data_LR[:limit, 1] * self.Factor
                else:
                    raise ValueError("外部导入仅支持 '时间谱/时间序列' 或 '空间谱'")

            elif self.type == '无不平顺':
                VL_Spline, VR_Spline, LL_Spline, LR_Spline = [np.zeros(FlowDimension) for _ in range(4)]
            else:
                raise ValueError(f"未知的激扰类型: {self.type}")

            # 统一前置无不平顺工况：
            # - 随机不平顺 / 外部导入 均在开头增加 self.Tstart 秒零激励，
            #   使列车先在平顺轨道上稳定运行。
            lead_steps = int(np.round(max(float(self.Tstart), 0.0) / self.Tstep))
            lead_zeros = np.zeros(lead_steps)

            # 2.1/2.2 平滑与前端缓冲：
            # - 随机谱/无不平顺：保留原逻辑，减小起步突变
            # - 外部导入：为避免“前置零激励 -> 外部序列”硬拼接导致边界冲击，
            #   做两步连续化处理：
            #   1) 首值去偏置（C0 连续）：x <- x - x[0]
            #   2) 短窗渐入（half-cosine ramp）：抑制边界导数尖峰
            if self.type == '外部导入':
                def _external_c0_ramp(sig: np.ndarray) -> np.ndarray:
                    s = np.asarray(sig, dtype=float).copy()
                    if s.size == 0:
                        return s

                    # 1) 首值去偏置，确保与前置零段位移连续
                    s -= float(s[0])

                    # 2) 短窗渐入（20ms），抑制边界处导数突变
                    ramp_time_s = 0.02
                    ramp_steps = int(np.round(ramp_time_s / self.Tstep))
                    ramp_steps = max(1, min(ramp_steps, s.size))
                    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp_steps))
                    s[:ramp_steps] *= ramp
                    return s

                VL_Spline = _external_c0_ramp(VL_Spline)
                VR_Spline = _external_c0_ramp(VR_Spline)
                LL_Spline = _external_c0_ramp(LL_Spline)
                LR_Spline = _external_c0_ramp(LR_Spline)

                bpsz_L = np.concatenate([lead_zeros, VL_Spline])
                bpsz_R = np.concatenate([lead_zeros, VR_Spline])
                bpsy_L = np.concatenate([lead_zeros, LL_Spline])
                bpsy_R = np.concatenate([lead_zeros, LR_Spline])
            else:
                SmoothStep = 8000
                SmoothDistance = 30
                if FlowDimension > SmoothStep:
                    SmoothFunction = np.concatenate([np.linspace(0, 1, SmoothStep), np.ones(FlowDimension - SmoothStep)])
                else:
                    SmoothFunction = np.linspace(0, 1, FlowDimension)

                VL_Spline *= SmoothFunction
                VR_Spline *= SmoothFunction
                LL_Spline *= SmoothFunction
                LR_Spline *= SmoothFunction

                pad_len = int(np.round((2 * (self.Lt + self.Lc) + SmoothDistance) / self.Vc / self.Tstep))
                pad_zeros = np.zeros(pad_len)

                bpsz_L = np.concatenate([lead_zeros, pad_zeros, VL_Spline])
                bpsz_R = np.concatenate([lead_zeros, pad_zeros, VR_Spline])
                bpsy_L = np.concatenate([lead_zeros, pad_zeros, LL_Spline])
                bpsy_R = np.concatenate([lead_zeros, pad_zeros, LR_Spline])

            # 2.3 数值差分计算激扰速度 (导数)
            dbpsz_L = np.append(np.diff(bpsz_L) / self.Tstep, 0.0)
            dbpsz_R = np.append(np.diff(bpsz_R) / self.Tstep, 0.0)
            dbpsy_L = np.append(np.diff(bpsy_L) / self.Tstep, 0.0)
            dbpsy_R = np.append(np.diff(bpsy_R) / self.Tstep, 0.0)

        # ==========================================
        # 3. 载荷维度屏蔽 (LoadType)
        # ==========================================
        if self.LoadType == 1:    # 仅施加横向
            bpsz_L = np.zeros_like(bpsz_L); dbpsz_L = np.zeros_like(dbpsz_L)
            bpsz_R = np.zeros_like(bpsz_R); dbpsz_R = np.zeros_like(dbpsz_R)
            # 根据原 MATLAB 设定：施加横向激扰时，横向速度必须为0
            dbpsy_L = np.zeros_like(dbpsy_L); dbpsy_R = np.zeros_like(dbpsy_R)  
        elif self.LoadType == 2:  # 仅施加垂向
            bpsy_L = np.zeros_like(bpsy_L); dbpsy_L = np.zeros_like(dbpsy_L)
            bpsy_R = np.zeros_like(bpsy_R); dbpsy_R = np.zeros_like(dbpsy_R)
        elif self.LoadType == 3:  # 垂向与横向双向耦合
            pass

        # ==========================================
        # 4. 多轮对相位延迟处理 (时空堆叠核心)
        # ==========================================
        def stack_wheelsets(arr):
            """将一维数组根据车辆轮对延迟切片，堆叠为 4 x (Nt+1) 的动力学标准输入矩阵"""
            # 增加越界安全保护，防止仿真时间过短导致数组越界
            end_idx = self.Nt + 1
            return np.vstack([
                arr[step1 : end_idx + step1],  # 轮对 1
                arr[step2 : end_idx + step2],  # 轮对 2
                arr[step3 : end_idx + step3],  # 轮对 3
                arr[0 : end_idx]               # 轮对 4 (基准)
            ])

        return (stack_wheelsets(bpsz_L), stack_wheelsets(bpsy_L), 
                stack_wheelsets(dbpsz_L), stack_wheelsets(dbpsy_L), 
                stack_wheelsets(bpsz_R), stack_wheelsets(bpsy_R), 
                stack_wheelsets(dbpsz_R), stack_wheelsets(dbpsy_R), 
                a, L)
    
    def excitation_harmonic(A: float, lanm: float, v: float, Tstep: float, Tz: float, Lstart: float):
        """
        生成谐波不平顺及其变化率 (采用高精度解析导数)
        输入:
            A: 谐波幅值 (m)
            lanm: 谐波波长 (m)
            v: 列车运行速度 (m/s)
            Tstep: 仿真积分步长 (s)
            Tz: 仿真总时间参数 (s)
            Lstart: 激扰空白启动时间 (s)
        输出:
            Irreg: 不平顺位移数组
            dIrreg: 不平顺速度(变化率)数组
        """
        t = np.arange(0, 1.5 * Tz + Tstep / 2.0, Tstep)
        # 核心物理量：时间圆频率
        omg = 2 * np.pi * v / lanm
        # 激扰启动前的平稳期
        pad_len = int(np.round(Lstart / Tstep))
        pad_zeros = np.zeros(pad_len)
        # 解析法计算激扰位移与速度
        Irreg = np.concatenate([pad_zeros, A * np.sin(omg * t)])
        dIrreg = np.concatenate([pad_zeros, A * omg * np.cos(omg * t)])
        return Irreg, dIrreg

    def irregularity_pad_generation(self, defect_switch: str = 'on'):
        """
        轨道不平顺生成 - 利用相干性推导左右轨实体不平顺，并叠加局部病害
        defect_switch: 是否添加冻胀/焊缝病害 ('on' 或 'off')
        返回: 
            Irre_GaoDi_L, Irre_GaoDi_R (N x 2 数组)
            Irre_GuiXiang_L, Irre_GuiXiang_R (N x 2 数组)
            a, L (病害参数)
        """
        # 调用类内部的 IFFT 方法，分别生成 N x 2 矩阵 (列0: 横坐标, 列1: 幅值)
        UM_Data_GaoDi = self.irregularity_generation('高低')
        UM_Data_ShuiPing = self.irregularity_generation('水平')
        UM_Data_GuiXiang = self.irregularity_generation('轨向')
        UM_Data_Gauge = self.irregularity_generation('轨距')
        # 提取横坐标
        X_axis = UM_Data_GaoDi[:, 0]

        # PART2. 生成冻胀/焊缝局部病害波形
        Z0, a, L = weld_bead(self.Vc, self.Tstep, defect_type='余弦函数_冻胀')
        Z0 = Z0 * 1000.0  # 将病害从米转换为毫米，以匹配 Irre_GaoDi_L_amp 的单位
        Z_len = len(Z0)
        
        # PART3. 左右钢轨相干性推导与病害随机叠加
        sample_rate = 1.0 / self.Tstep
        # 计算随机叠加的位置 (在 0.5 秒到 2.0 秒之间随机注入)
        random_decimal = np.random.rand() * 1.5 + 0.5
        overlay_position = int(np.round(random_decimal * sample_rate))

        # SEC1. 高低不平顺的左右相干性
        # 公式: 左高低 = 高低 + 0.5 * 水平 ; 右高低 = 高低 - 0.5 * 水平
        Irre_GaoDi_L_amp = UM_Data_GaoDi[:, 1] + 0.5 * UM_Data_ShuiPing[:, 1]
        Irre_GaoDi_R_amp = UM_Data_GaoDi[:, 1] - 0.5 * UM_Data_ShuiPing[:, 1]
        if defect_switch.lower() == 'on':
            end_pos = overlay_position + Z_len
            # 安全越界保护机制：防止随机生成的位置太靠后导致数组越界
            if end_pos <= len(Irre_GaoDi_L_amp):
                Irre_GaoDi_L_amp[overlay_position:end_pos] += Z0
                Irre_GaoDi_R_amp[overlay_position:end_pos] += Z0
            else:
                print(" -> [警告] 随机病害叠加位置越界，已进行截断处理！")
                valid_len = len(Irre_GaoDi_L_amp) - overlay_position
                Irre_GaoDi_L_amp[overlay_position:] += Z0[:valid_len]
                Irre_GaoDi_R_amp[overlay_position:] += Z0[:valid_len]
        else:
            print(" -> [提示] 未添加局部病害")
            a, L = 0, 0
        # SEC2. 轨向不平顺的左右相干性
        # 公式: 左轨向 = 轨向 + 0.5 * 轨距 ; 右轨向 = 轨向 - 0.5 * 轨距
        Irre_GuiXiang_L_amp = UM_Data_GuiXiang[:, 1] + 0.5 * UM_Data_Gauge[:, 1]
        Irre_GuiXiang_R_amp = UM_Data_GuiXiang[:, 1] - 0.5 * UM_Data_Gauge[:, 1]

        # SEC3. 重新组装为 N x 2 的矩阵返回
        Irre_GaoDi_L = np.column_stack((X_axis, Irre_GaoDi_L_amp))
        Irre_GaoDi_R = np.column_stack((X_axis, Irre_GaoDi_R_amp))
        Irre_GuiXiang_L = np.column_stack((X_axis, Irre_GuiXiang_L_amp))
        Irre_GuiXiang_R = np.column_stack((X_axis, Irre_GuiXiang_R_amp))

        return Irre_GaoDi_L, Irre_GaoDi_R, Irre_GuiXiang_L, Irre_GuiXiang_R, a, L
    
    def irregularity_generation(self,
                                irr_type: str):
        """
        核心不平顺样本生成 (IFFT 法)
        irr_type: 传入需要生成的不平顺类型 (如 '高低', '水平', '轨向', '轨距')
        返回: UM_Data 格式的 N x 2 数组 (列0: 时间, 列1: 不平顺幅值 mm)
        """
        Tc = self.Tstep
        t = self.mile / self.Vc
        f_min = self.Vc / self.lanm_max          #最小频率
        f_max = self.Vc / self.lanm_min          #最大频率
        f_1 = self.Vc / self.Factor              #频率转换区域 (极高频阈值)

        N = t/Tc                                 #时间采样数据量
        N_r = 2 ** int(np.ceil(np.log2(N)))      #满足采样定理的最小数据量
        delta_f = 1 / (N_r * Tc)                 #频率分辨率
        N_f = int(np.round(f_max / delta_f))     #最高频率所在索引
        N_1 = int(np.round(f_1 / delta_f))       #频率转换阈值所在索引
        N_0 = int(np.round(f_min / delta_f))     #最低频率所在索引

        # 1. 向量化生成空间频率域
        freq_indices = np.arange(N_0, N_f + 1)  
        f = freq_indices * delta_f                 #对应频率值
        space_frequency = f / self.Vc              #空间频率
        space_wave_length = 2 * np.pi * space_frequency    #空间波长
        S_k = self._get_power_spectrum(space_frequency, space_wave_length, f, f_1, irr_type)

        # 2. IFFT 逆变换组装核心
        phi = 2 * np.pi * np.random.rand(len(S_k))
        epsilon = np.exp(1j * phi)
        b = N_r * epsilon * np.sqrt(S_k * delta_f)
        e = np.zeros(N_r, dtype=complex)
        e[N_0 : N_f + 1] = b
        e[N_r - N_f : N_r - N_0 + 1] = np.conj(b[::-1])
        g = np.real(np.fft.ifft(e))
        Time = np.arange(1, len(g) + 1) * Tc
        UM_Data = np.column_stack((Time, 10.0 * g))
        return UM_Data
    
    
    def _get_power_spectrum(self,
                            space_frequency, 
                            space_wave_length,
                            f_array,
                            f_1, 
                            irr_type: str
                            ):
        """
        根据设定的功率谱类型计算功率谱值
        返回: 功率谱值数组 S_k
        """
        # 1. 统一输出变量名为 S_k (大写)
        S_k = np.zeros_like(space_frequency)
        
        # 2. 修正掩码逻辑：必须用时间频率 (f_array) 去和 f_1 比较！
        mask_low = f_array < f_1
        mask_high = f_array >= f_1

        if self.powerSpectrum_type == '高铁谱':
            raw_S = PowerSpectrum_ChineseHighSpeed(space_frequency, irr_type)
            # 补上缺失的量纲缩放
            S_k = raw_S / self.Vc / 100.0
            
        elif self.powerSpectrum_type in ['常规', '常规谱']: # 兼容两种写法
            omega1 = space_wave_length[mask_low]
            S_k[mask_low] = 4 * 0.25 * 0.0339 * (0.8245**2) / ((omega1**2 + 0.438**2) * (omega1**2 + 0.8245**2)) * 2 * np.pi / self.Vc / 2
            
            omega2 = space_frequency[mask_high]
            S_k[mask_high] = 0.036 * omega2**(-3.15) / self.Vc / 100.0 / 2.0
            
        elif self.powerSpectrum_type == '干线谱':
            raw_S = PSD_China_GanXian(space_frequency, irr_type)
            S_k = raw_S / self.Vc / 100.0
            
        elif self.powerSpectrum_type == '美国谱':
            raw_S = PSD_America(space_frequency, irr_type, level=6)
            S_k[mask_low] = raw_S[mask_low] * 1e4 / self.Vc
            # 原版 MATLAB 中大于 f_1 的部分被注释掉了，且沿用了原始公式，所以直接全频段覆盖也可以，这里忠实原版逻辑
            S_k[mask_high] = raw_S[mask_high] * 1e4 / self.Vc 
            
        elif self.powerSpectrum_type == '德国低干扰谱':
            raw_S = PowerSpectrum_German(space_wave_length, irr_type, interference='低干扰')
            S_k = raw_S / self.Vc / 100.0
            
        elif self.powerSpectrum_type == '德国高干扰谱':
            raw_S = PowerSpectrum_German(space_wave_length, irr_type, interference='高干扰')
            S_k = raw_S / self.Vc / 100.0
            # 德国高干扰谱存在高频强截断衰减
            S_k[mask_high] = 0.036 * space_frequency[mask_high]**(-3.15) / self.Vc / 100.0 / 2.0
            
        else:
            raise ValueError(f"未知的功率谱类型: {self.powerSpectrum_type}")
        
        # 统一返回 S_k
        return S_k