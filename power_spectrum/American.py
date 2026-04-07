'''
Author: Niscienc 60505912+2099193635@users.noreply.github.com
Date: 2026-03-11 16:25:39
LastEditors: Niscienc 60505912+2099193635@users.noreply.github.com
LastEditTime: 2026-03-11 16:26:57
FilePath: \VTCM_PYTHON\power_spectrum\American.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np

def PSD_America(SpaceFrequency, irr_type: str, level: int = 6) -> np.ndarray:
    """
    美国标准轨道功率谱计算
    输入:
        SpaceFrequency: 空间频率数组 (1/m)
        irr_type: 不平顺类型 ('高低', '轨向', '水平', '轨距')
        level: 美国轨道等级 (1-6 级，数字越大路况越好，默认 6 级)
    输出:
        S: 功率谱密度数组 (通常为 cm^2 * rad / m)
    """
    Phi = np.asarray(SpaceFrequency)
    # 物理保护：美国谱高低/轨向分母有 Phi^4，防止 0 频率导致的无限大爆炸
    Phi_safe = np.where(Phi == 0, 1e-10, Phi)

    # 校验等级 (1-6)
    if not (1 <= level <= 6):
        raise ValueError("美国谱的等级 (Level) 必须在 1 到 6 之间。")
    idx = level - 1  # 转换为 0-based 

    # --- 1. 参数矩阵定义 ---
    Av_arr = np.array([16.7217, 9.5250, 5.2917, 2.9633, 1.6722, 0.9525]) * 1e-7
    Phiv1_arr = np.array([0.0233, 0.0233, 0.0233, 0.0233, 0.0233, 0.0233])
    Phiv2_arr = np.array([0.1312, 0.1312, 0.1312, 0.1312, 0.1312, 0.1312])

    Aa_arr = np.array([10.5833, 5.9267, 3.3867, 1.8838, 1.0583, 0.5927]) * 1e-7
    Phia1_arr = np.array([0.0328, 0.0328, 0.0328, 0.0328, 0.0328, 0.0328])
    Phia2_arr = np.array([0.1837, 0.1837, 0.1837, 0.1837, 0.1837, 0.1837])

    Ac_arr = np.array([4.8683, 3.3867, 2.3283, 1.5663, 1.0583, 0.7197]) * 1e-7
    Phic1_arr = np.array([0.0233, 0.0233, 0.0233, 0.0233, 0.0233, 0.0233])
    Phic2_arr = np.array([0.1312, 0.1312, 0.1312, 0.1312, 0.1312, 0.1312])

    Ag_arr = np.array([10.5833, 5.9267, 3.3867, 1.8838, 1.0583, 0.5927]) * 1e-7
    Phig1_arr = np.array([0.0292, 0.0292, 0.0292, 0.0292, 0.0292, 0.0292])
    Phig2_arr = np.array([0.2329, 0.2329, 0.2329, 0.2329, 0.2329, 0.2329])

    # --- 2. 向量化提取与计算 ---
    if irr_type in ['高低', '左轨高低', '右轨高低', '左高低', '右高低']:
        Av, Phiv1, Phiv2 = Av_arr[idx], Phiv1_arr[idx], Phiv2_arr[idx]
        S = Av * (Phiv2**2) * (Phi_safe**2 + Phiv1**2) / (Phi_safe**4 * (Phi_safe**2 + Phiv2**2))
        
    elif irr_type in ['轨向', '左轨轨向', '右轨轨向', '左轨向', '右轨向']:
        Aa, Phia1, Phia2 = Aa_arr[idx], Phia1_arr[idx], Phia2_arr[idx]
        S = Aa * (Phia2**2) * (Phi_safe**2 + Phia1**2) / (Phi_safe**4 * (Phi_safe**2 + Phia2**2))
        
    elif irr_type == '水平':
        Ac, Phic1, Phic2 = Ac_arr[idx], Phic1_arr[idx], Phic2_arr[idx]
        # 水平公式分母没有单一的 Phi^4，不用担心爆炸，但用 Phi_safe 依旧更鲁棒
        S = Ac * (Phic2**2) / ((Phi_safe**2 + Phic1**2) * (Phi_safe**2 + Phic2**2))
        
    elif irr_type == '轨距':
        Ag, Phig1, Phig2 = Ag_arr[idx], Phig1_arr[idx], Phig2_arr[idx]
        S = Ag * (Phig2**2) / ((Phi_safe**2 + Phig1**2) * (Phi_safe**2 + Phig2**2))
        
    else:
        raise ValueError(f"美国谱不支持的不平顺类型: {irr_type}")

    return S

if __name__ == "__main__":
    # 测试美国谱计算
    freqs = np.array([0.01, 0.1, 1, 10])  # 空间频率 (1/m)
    irr_type = '高低'
    S_values = PSD_America(freqs, irr_type, level=6)
    print("空间频率 (1/m):", freqs)
    print("美国谱功率谱密度 (cm^2 * rad / m):", S_values)