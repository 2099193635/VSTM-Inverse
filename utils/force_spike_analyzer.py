#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨道车辆力突起分析工具 (Force Spike Analysis Tool)
用于分析轨道不规则性与车轮动态力之间的关系
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def analyze_force_spike(result_npz_path, output_dir=None, verbose=True):
    """
    分析仿真结果中的力突起现象
    
    Parameters:
    -----------
    result_npz_path : str
        仿真结果.npz文件路径
    output_dir : str, optional
        输出目录，默认为result_npz_path的上级目录
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    dict : 分析结果字典
    """
    
    # 加载数据
    if verbose:
        print(f"Loading simulation data from: {result_npz_path}")
    
    data = np.load(result_npz_path)
    dt = float(data['dt'])
    total_steps = len(data['Track_abs_mileage_m'])
    time_array = np.arange(total_steps) * dt
    
    # 获取关键数据
    Irre_dbz_L = data['Irre_dbz_L_ref']
    Irre_dbz_R = data['Irre_dbz_R_ref']
    Irre_bz_L = data['Irre_bz_L_ref']
    Irre_bz_R = data['Irre_bz_R_ref']
    TotalVerticalForce = data['TotalVerticalForce']
    TotalLateralForce = data['TotalLateralForce']
    distance_m = data['Irre_distance_m']
    
    # 计算高程的二阶导数（曲率）
    Irre_d2bz_L = np.diff(Irre_dbz_L)
    Irre_d2bz_R = np.diff(Irre_dbz_R)
    
    # 计算力的变化率
    dF_dt = np.diff(TotalVerticalForce, axis=0) / dt
    dFy_dt = np.diff(TotalLateralForce, axis=0) / dt
    
    # 构建结果字典
    results = {
        'dt': dt,
        'total_steps': total_steps,
        'total_time': total_steps * dt,
        'time_array': time_array,
        'distance_m': distance_m,
        
        # 高程特性
        'elevation': {
            'left_min': Irre_bz_L.min(),
            'left_max': Irre_bz_L.max(),
            'left_std': np.std(Irre_bz_L),
            'right_min': Irre_bz_R.min(),
            'right_max': Irre_bz_R.max(),
            'right_std': np.std(Irre_bz_R),
        },
        
        # 斜率特性
        'slope': {
            'left_min': Irre_dbz_L.min(),
            'left_max': Irre_dbz_L.max(),
            'left_max_abs': np.max(np.abs(Irre_dbz_L)),
            'right_min': Irre_dbz_R.min(),
            'right_max': Irre_dbz_R.max(),
            'right_max_abs': np.max(np.abs(Irre_dbz_R)),
        },
        
        # 曲率特性（最关键）
        'curvature': {
            'left_min': np.min(Irre_d2bz_L),
            'left_max': np.max(Irre_d2bz_L),
            'left_max_abs': np.max(np.abs(Irre_d2bz_L)),
            'left_peak_idx': np.argmax(np.abs(Irre_d2bz_L)),
            'left_peak_distance': distance_m[np.argmax(np.abs(Irre_d2bz_L))],
            'right_min': np.min(Irre_d2bz_R),
            'right_max': np.max(Irre_d2bz_R),
            'right_max_abs': np.max(np.abs(Irre_d2bz_R)),
            'right_peak_idx': np.argmax(np.abs(Irre_d2bz_R)),
            'right_peak_distance': distance_m[np.argmax(np.abs(Irre_d2bz_R))],
        },
        
        # 力的特性
        'forces': {},
    }
    
    # 逐轮分析力
    for wheel_id in range(8):
        force = TotalVerticalForce[:, wheel_id]
        force_rate = dF_dt[:, wheel_id]
        lateral_force = TotalLateralForce[:, wheel_id]
        lateral_rate = dFy_dt[:, wheel_id]
        
        max_rate_idx = np.argmax(np.abs(force_rate))
        max_lateral_idx = np.argmax(np.abs(lateral_rate))
        
        results['forces'][wheel_id] = {
            'vertical_mean': np.mean(force),
            'vertical_max': np.max(force),
            'vertical_min': np.min(force),
            'vertical_std': np.std(force),
            'vertical_max_rate': np.max(np.abs(force_rate)),
            'vertical_max_rate_time': time_array[max_rate_idx],
            'vertical_max_rate_idx': max_rate_idx,
            
            'lateral_mean': np.mean(lateral_force),
            'lateral_max': np.max(lateral_force),
            'lateral_min': np.min(lateral_force),
            'lateral_max_rate': np.max(np.abs(lateral_rate)),
            'lateral_max_rate_time': time_array[max_lateral_idx],
        }
    
    # 打印报告
    if verbose:
        print_analysis_report(results)
    
    # 生成图表
    if output_dir is None:
        output_dir = str(Path(result_npz_path).parent.parent / 'figures')
    
    os.makedirs(output_dir, exist_ok=True)
    plot_analysis(results, TotalVerticalForce, TotalLateralForce, 
                  Irre_bz_L, Irre_bz_R, Irre_dbz_L, Irre_dbz_R,
                  Irre_d2bz_L, Irre_d2bz_R, output_dir, verbose=verbose)
    
    return results


def print_analysis_report(results):
    """打印分析报告"""
    print("\n" + "="*80)
    print("FORCE SPIKE ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nSimulation Duration: {results['total_time']:.2f} s")
    print(f"Total Steps: {results['total_steps']}")
    print(f"Time Step: {results['dt']*1000:.4f} ms")
    
    print("\n--- Track Elevation Profile ---")
    elev = results['elevation']
    print(f"Left Rail:  [{elev['left_min']*1000:+.3f}, {elev['left_max']*1000:+.3f}] mm, σ={elev['left_std']*1000:.3f} mm")
    print(f"Right Rail: [{elev['right_min']*1000:+.3f}, {elev['right_max']*1000:+.3f}] mm, σ={elev['right_std']*1000:.3f} mm")
    
    print("\n--- Track Slope ---")
    slope = results['slope']
    print(f"Left Rail:  [{slope['left_min']:+.4f}, {slope['left_max']:+.4f}], max|slope|={slope['left_max_abs']:.4f}")
    print(f"Right Rail: [{slope['right_min']:+.4f}, {slope['right_max']:+.4f}], max|slope|={slope['right_max_abs']:.4f}")
    
    print("\n--- CRITICAL: Track Curvature (d²z/dx²) ---")
    curv = results['curvature']
    print(f"Left Rail Curvature:")
    print(f"    Range: [{curv['left_min']:+.6f}, {curv['left_max']:+.6f}] m⁻¹")
    print(f"    Max Abs: {curv['left_max_abs']:.6f} m⁻¹ at {curv['left_peak_distance']:.2f}m")
    print(f"Right Rail Curvature:")
    print(f"    Range: [{curv['right_min']:+.6f}, {curv['right_max']:+.6f}] m⁻¹")
    print(f"    Max Abs: {curv['right_max_abs']:.6f} m⁻¹ at {curv['right_peak_distance']:.2f}m")
    
    print("\n--- Vertical Force Analysis (Per Wheel) ---")
    print("Wheel | Mean(kN) | Max(kN) | Max Rate(MN/s) | Peak Time(s)")
    print("------|----------|---------|----------------|-------------")
    for wheel_id in range(8):
        f = results['forces'][wheel_id]
        print(f"{wheel_id:5d} | {f['vertical_mean']/1000:8.1f} | {f['vertical_max']/1000:7.1f} | "
              f"{f['vertical_max_rate']/1000000:14.2f} | {f['vertical_max_rate_time']:13.3f}")
    
    # 前后轮比较
    front_max = max(results['forces'][i]['vertical_max'] for i in [0,1,2,3])
    rear_max = max(results['forces'][i]['vertical_max'] for i in [4,5,6,7])
    amplification = rear_max / front_max
    print(f"\n--- Amplification Analysis ---")
    print(f"Front Wheels Max Force: {front_max/1000000:.2f} MN")
    print(f"Rear Wheels Max Force:  {rear_max/1000000:.2f} MN")
    print(f"Amplification Factor:   {amplification:.1f}x")


def plot_analysis(results, TotalVerticalForce, TotalLateralForce,
                  Irre_bz_L, Irre_bz_R, Irre_dbz_L, Irre_dbz_R,
                  Irre_d2bz_L, Irre_d2bz_R, output_dir, verbose=False):
    """生成分析图表"""
    
    time_array = results['time_array']
    distance_m = results['distance_m']
    dt = results['dt']
    dF_dt = np.diff(TotalVerticalForce, axis=0) / dt
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 图1: 高程
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(distance_m, Irre_bz_L * 1000, label='Left Rail', linewidth=0.8, alpha=0.7)
    ax1.plot(distance_m, Irre_bz_R * 1000, label='Right Rail', linewidth=0.8, alpha=0.7)
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Elevation (mm)')
    ax1.set_title('Track Elevation Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 斜率
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(distance_m, Irre_dbz_L, label='Left Rail', linewidth=0.8, alpha=0.7)
    ax2.plot(distance_m, Irre_dbz_R, label='Right Rail', linewidth=0.8, alpha=0.7)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Slope (dz/dx)')
    ax2.set_title('Track Slope')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 曲率（最重要）
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(distance_m[:-1], Irre_d2bz_L, label='Left Rail', linewidth=0.8, alpha=0.7, color='blue')
    ax3.plot(distance_m[:-1], Irre_d2bz_R, label='Right Rail', linewidth=0.8, alpha=0.7, color='red')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Curvature (m⁻¹)')
    ax3.set_title('Track Curvature (d²z/dx²) - KEY DRIVER OF FORCE SPIKE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 竖向力时间序列
    ax4 = fig.add_subplot(gs[1, 1])
    for wheel_id in [0, 1, 4, 5]:
        label = f'Front-{wheel_id}' if wheel_id < 2 else f'Rear-{wheel_id}'
        ax4.plot(time_array, TotalVerticalForce[:, wheel_id]/1000, 
                label=label, linewidth=0.6, alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Vertical Force (kN)')
    ax4.set_title('Vertical Forces vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 图5: 力变化率
    ax5 = fig.add_subplot(gs[2, 0])
    for wheel_id in [0, 1, 4, 5]:
        label = f'Front-{wheel_id}' if wheel_id < 2 else f'Rear-{wheel_id}'
        ax5.plot(time_array[:-1], np.abs(dF_dt[:, wheel_id])/1000000,
                label=label, linewidth=0.6, alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('|dF/dt| (MN/s)')
    ax5.set_title('Magnitude of Force Rate of Change')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 图6: 摘要统计
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    curv = results['curvature']
    summary_text = f"""
KEY FINDINGS

Track Defects:
  • Max curvature: {curv['left_max_abs']:.2f} m⁻¹ (Left)
  • Defect location: {curv['left_peak_distance']:.1f}m

Force Spikes:
  • Front wheel max: {max(results['forces'][i]['vertical_max'] for i in range(4))/1000000:.2f} MN
  • Rear wheel max:  {max(results['forces'][i]['vertical_max'] for i in range(4,8))/1000000:.2f} MN
  • Amplification:   {max(results['forces'][i]['vertical_max'] for i in range(4,8)) / max(results['forces'][i]['vertical_max'] for i in range(4)):.1f}x

Peak Force Rate:
  • Front wheels: {max(results['forces'][i]['vertical_max_rate'] for i in range(4))/1000000:.2f} MN/s
  • Rear wheels:  {max(results['forces'][i]['vertical_max_rate'] for i in range(4,8))/1000000:.2f} MN/s

Root Cause:
  Sharp track geometry curvature causes
  impulsive wheel loading, amplified by
  suspension resonance on rear wheels.
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 保存图表
    output_path = os.path.join(output_dir, 'force_spike_detailed_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"Analysis plot saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    # 示例用法
    result_file = r'c:\VTCM_PYTHON\results\default_project\高速客车-外部导入-vehicle-standard-20260319_101158\files\simulation_result.npz'
    output_dir = r'c:\VTCM_PYTHON\results\default_project\高速客车-外部导入-vehicle-standard-20260319_101158\figures'
    
    results = analyze_force_spike(result_file, output_dir=output_dir, verbose=True)
    print("\nAnalysis complete!")
