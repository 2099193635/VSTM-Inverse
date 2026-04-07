import numpy as np

def PowerSpectrum_ChineseHighSpeed(SpaceFrequency, irr_type: str) -> np.ndarray:
    """
    中国高速铁路轨道功率谱值计算
    输入:
        SpaceFrequency: 空间频率数组 (1/m)
        irr_type: 不平顺类型 ('高低', '轨向', '水平', '轨距')
    输出:
        S: 功率谱密度数组 (mm^2 / (1/m))
    """
    SpaceFrequency = np.asarray(SpaceFrequency)
    S = np.zeros_like(SpaceFrequency, dtype=float)
    # 物理保护：防止低频段出现 0 导致 0^(-n) 除零报错，赋一个极小值
    sf_safe = np.where(SpaceFrequency == 0, 1e-10, SpaceFrequency)
    if irr_type == '高低':
        sw = [0.0187, 0.0474, 0.1533]
        A = [1.0544e-5, 3.5588e-3, 1.9784e-2, 3.9488e-4]
        n = [3.3891, 1.9271, 1.3643, 3.4516]
        # 生成分段掩码
        m1 = sf_safe < sw[0]
        m2 = (sf_safe >= sw[0]) & (sf_safe < sw[1])
        m3 = (sf_safe >= sw[1]) & (sf_safe < sw[2])
        m4 = sf_safe >= sw[2]
        # 向量化分段赋值
        S[m1] = A[0] / (sf_safe[m1] ** n[0])
        S[m2] = A[1] / (sf_safe[m2] ** n[1])
        S[m3] = A[2] / (sf_safe[m3] ** n[2])
        S[m4] = A[3] / (sf_safe[m4] ** n[3])

    elif irr_type == '轨向':
        sw = [0.045, 0.1234]
        A = [3.9513e-3, 1.1047e-2, 7.5633e-4]
        n = [1.867, 1.5354, 2.8171]

        m1 = sf_safe < sw[0]
        m2 = (sf_safe >= sw[0]) & (sf_safe < sw[1])
        m3 = sf_safe >= sw[1]

        S[m1] = A[0] / (sf_safe[m1] ** n[0])
        S[m2] = A[1] / (sf_safe[m2] ** n[1])
        S[m3] = A[2] / (sf_safe[m3] ** n[2])

    elif irr_type == '水平':
        sw = [0.0258, 0.1163]
        A = [3.6148e-3, 4.3685e-2, 4.5867e-3]
        n = [1.7278, 1.0461, 2.0939]

        m1 = sf_safe < sw[0]
        m2 = (sf_safe >= sw[0]) & (sf_safe < sw[1])
        m3 = sf_safe >= sw[1]

        S[m1] = A[0] / (sf_safe[m1] ** n[0])
        S[m2] = A[1] / (sf_safe[m2] ** n[1])
        S[m3] = A[2] / (sf_safe[m3] ** n[2])

    elif irr_type == '轨距':
        sw = [0.1090, 0.2938]
        A = [5.4978e-2, 5.0701e-3, 1.8778e-4]
        n = [0.8282, 1.9037, 4.5948]

        m1 = sf_safe < sw[0]
        m2 = (sf_safe >= sw[0]) & (sf_safe < sw[1])
        m3 = sf_safe >= sw[1]

        S[m1] = A[0] / (sf_safe[m1] ** n[0])
        S[m2] = A[1] / (sf_safe[m2] ** n[1])
        S[m3] = A[2] / (sf_safe[m3] ** n[2])
    
    else:
        raise ValueError(f"未知的不平顺类型: {irr_type}")

    return S

def PSD_China_GanXian(SpaceFrequency, irr_type) -> np.ndarray:
    """
    中国既有铁路提速干线轨道功率谱计算 (NumPy 极速向量化版)
    输入:
        SpaceFrequency: 空间频率数组 (1/m)
        irr_type: 不平顺类型 (支持数字 1-5，或直观的字符串如 '高低', '水平')
    输出:
        S: 功率谱密度数组 (mm^2 / (1/m))
    """
    f = np.asarray(SpaceFrequency)
    
    # --- 1. 参数矩阵 ---
    A_arr = np.array([1.1029, 0.8581, 0.2244, 0.3743, 0.1214])
    B_arr = np.array([-1.4709, -1.4607, -1.5746, -1.5894, -2.1603])
    C_arr = np.array([0.5941, 0.5848, 0.6683, 0.7265, 2.0214])
    D_arr = np.array([0.848, 0.0407, -2.1466, 0.4353, 4.5089])
    E_arr = np.array([3.8016, 2.8428, 1.7665, 0.9101, 2.2227])
    F_arr = np.array([-0.25, -0.1989, -0.1506, -0.027, -0.0396])
    G_arr = np.array([0.0112, 0.0094, 0.0052, 0.0031, 0.0073])

    # --- 2. 智能解析类型索引 ---
    idx = 0
    if isinstance(irr_type, int):
        idx = irr_type - 1  
    else:
        mapping = {
            '左轨高低': 0, '左高低': 0,
            '右轨高低': 1, '右高低': 1,
            '左轨轨向': 2, '左轨向': 2,
            '右轨轨向': 3, '右轨向': 3,
            '水平': 4
        }
        if irr_type in mapping:
            idx = mapping[irr_type]
        else:
            raise ValueError(f"未知的干线谱不平顺类型: {irr_type}")

    # --- 3. 提取对应维度的 7 个系数 ---
    A, B, C = A_arr[idx], B_arr[idx], C_arr[idx]
    D, E, F_val, G = D_arr[idx], E_arr[idx], F_arr[idx], G_arr[idx]

    # --- 4. 向量化计算功率谱密度 ---
    numerator = f**2 + B * f + C
    denominator = f**4 + D * f**3 + E * f**2 + F_val * f + G
    S = A * numerator / denominator
    return S

if __name__ == "__main__":
    # 测试示例
    freqs = np.array([0.01, 0.05, 0.1, 0.5, 1.0])  # 空间频率 (1/m)
    irr_type = '高低'
    S_values = PowerSpectrum_ChineseHighSpeed(freqs, irr_type)
    print("空间频率 (1/m):", freqs)
    print("功率谱密度 (mm^2/(1/m)):", S_values)
    irr_type = '左轨高低'
    S_values_gx = PSD_China_GanXian(freqs, irr_type)
    print("干线谱功率谱密度 (mm^2/(1/m)):", S_values_gx)