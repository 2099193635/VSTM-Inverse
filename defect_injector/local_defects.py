'''
Author: Niscienc 60505912+2099193635@users.noreply.github.com
Date: 2026-03-11 19:56:58
LastEditors: Niscienc 60505912+2099193635@users.noreply.github.com
LastEditTime: 2026-03-11 19:58:43
FilePath: \VTCM_PYTHON\defect_injector\local_defects.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np

def weld_bead(Vc: float, Tstep: float, defect_type: str = '余弦函数_冻胀'):
    """
    生成局部冻胀波形/焊缝病害)
    输入:
        Vc: 列车速度 (m/s)
        Tstep: 仿真积分步长 (s)
        defect_type: 病害类型 ('余弦函数_冻胀' 或 '叠合不平顺_焊缝')
    返回: 
        Z0 (波形数组, 单位 m), a (幅值), L (波长)
    """
    v = Vc
    
    if defect_type == '余弦函数_冻胀':
        a = 5e-3                                                    # 冻胀幅值 (m)
        L = 5.0                                                     # 冻胀波长 (m)
        time_total = L / v
        
        # 使用 Tstep/2 作为安全裕度，防止浮点精度问题丢失最后一个点
        t = np.arange(0, time_total + Tstep / 2.0, Tstep) 
        
        # 向量化直接计算余弦波
        Z0 = 0.5 * a * (1 - np.cos(2 * np.pi * v * t / L))
        
    elif defect_type == '叠合不平顺_焊缝':
        L = 1.0         # 主波长 (m)
        lam = 0.1       # 短波波长 (m)
        a1 = -1e-3      # 主波深 (m)
        a2 = -0.05e-3   # 短波深 (m)
        time_total = L / v
        
        t = np.arange(0, time_total + Tstep / 2.0, Tstep)
        Z0 = np.zeros_like(t)
        
        # 计算分段的时间节点
        t0 = (L - lam) / (2 * v)
        t1 = (L + lam) / (2 * v)
        
        # 向量化分段掩码
        mask1 = t < t0
        mask2 = (t >= t0) & (t <= t1)
        mask3 = t > t1
        
        # 极速分段赋值
        Z0[mask1] = 0.5 * a1 * (1 - np.cos(2 * np.pi * v * t[mask1] / L))
        Z0[mask2] = a1 + 0.5 * a2 * (1 - np.cos(2 * np.pi * v * (t[mask2] - t0) / lam))
        Z0[mask3] = 0.5 * a1 * (1 - np.cos(2 * np.pi * v * t[mask3] / L))
        
    else:
        raise ValueError(f"未知的局部病害类型: {defect_type}")
        
    return Z0, a, L