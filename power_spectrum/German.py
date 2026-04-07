import numpy as np

def PowerSpectrum_German(Omega, irr_type: str, interference: str = '低干扰') -> np.ndarray:
    """
    德国轨道功率谱计算 (NumPy 极速向量化版, 统合高/低干扰)
    输入:
        Omega: 空间角频率数组 (rad/m)
        irr_type: 不平顺类型 ('高低', '轨向', '水平', '轨距')
        interference: '高干扰' 或 '低干扰' (默认 '低干扰')
    输出:
        S: 功率谱密度数组 (mm^2 / (rad/m))
    """
    Omega_safe = np.asarray(Omega)
    # 防止极个别情况下的除零
    Omega_safe = np.where(Omega_safe == 0, 1e-10, Omega_safe)

    # --- 1. 提取共享常量 ---
    Omega_c = 0.8246
    Omega_r = 0.0206
    Omega_s = 0.438
    b = 0.75

    # --- 2. 根据干扰类型自动分发系数 ---
    if interference == '高干扰':
        A_a = 6.125e-7
        A_v = 1.08e-6
        A_g = 1.032e-7
    elif interference == '低干扰':
        A_a = 2.119e-7
        A_v = 4.032e-7
        A_g = 5.32e-8
    else:
        raise ValueError(f"未知的德国谱干扰等级: {interference}")

    # --- 3. 计算核心公式 ---
    if irr_type in ['高低', '左轨高低', '右轨高低', '左高低', '右高低']:
        S = A_v * (Omega_c**2) / ((Omega_safe**2 + Omega_r**2) * (Omega_safe**2 + Omega_c**2))
    elif irr_type in ['轨向', '左轨轨向', '右轨轨向', '左轨向', '右轨向']:
        S = A_a * (Omega_c**2) / ((Omega_safe**2 + Omega_r**2) * (Omega_safe**2 + Omega_c**2))
    elif irr_type == '水平':
        S = (A_v * (b**-2) * (Omega_c**2) * (Omega_safe**2)) / \
            ((Omega_safe**2 + Omega_r**2) * (Omega_safe**2 + Omega_c**2) * (Omega_safe**2 + Omega_s**2))
    elif irr_type == '轨距':
        S = (A_g * (Omega_c**2) * (Omega_safe**2)) / \
            ((Omega_safe**2 + Omega_r**2) * (Omega_safe**2 + Omega_c**2) * (Omega_safe**2 + Omega_s**2))
            
    else:
        raise ValueError(f"德国谱不支持的不平顺类型: {irr_type}")

    return S * 1e6