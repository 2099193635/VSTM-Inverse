'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-03-24 06:40:03
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-03-26 11:19:17
FilePath: /VTCM_PYTHON/PINO/generate_dataset.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
#!/usr/bin/env python
"""
VTCM PINO 数据生成脚本

根据不同的起始里程位置生成多个仿真片段用于训练和测试集，支持参数扫描以增加数据多样性。
"""

import os
import sys
import subprocess
import argparse
import itertools
from pathlib import Path
from datetime import datetime


def _count_completed_runs(project_name: str) -> int:
    """统计指定项目下已完成的仿真数量（以 simulation_result.npz 为准）"""
    project_root = Path("/workspace/VTCM_PYTHON") / "results" / project_name
    if not project_root.exists():
        return 0
    return len(list(project_root.glob("*/files/simulation_result.npz")))


def generate_simulations(
    num_train: int = 8,
    num_test: int = 2,
    vehicle_types: list = None,
    irr_types: list = None,
    mileage_range: tuple = (270.0, 280.0),
    train_project: str = "pino_train_dataset",
    test_project: str = "pino_test_dataset",
    tz: float = 5.0,
    timeout: int = 12000,
    show_main_output: bool = True,
    resume: bool = True,
):
    """
    根据不同的起始里程位置生成训练和测试仿真片段
    
    Args:
        num_train: 训练集仿真次数
        num_test: 测试集仿真次数
        vehicle_types: 车辆类型列表
        irr_types: 不平顺类型列表
        mileage_range: 起始里程范围 (km) (min, max)
        train_project: 训练集输出项目名
        test_project: 测试集输出项目名
    """
    
    if vehicle_types is None:
        vehicle_types = ["高速客车"]
    
    if irr_types is None:
        irr_types = ["随机不平顺"]
    
    # 创建配置组合
    configs = list(itertools.product(vehicle_types, irr_types))
    
    print("\n" + "=" * 70)
    print("VTCM PINO 数据生成工具 - 里程参数化数据集构建")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置信息:")
    print(f"  训练集: {num_train:3d} 个仿真 → results/{train_project}/")
    print(f"  测试集: {num_test:3d} 个仿真 → results/{test_project}/")
    print(f"  里程范围: {mileage_range[0]:.3f} - {mileage_range[1]:.3f} km")
    print(f"  车辆类型: {', '.join(vehicle_types)}")
    print(f"  激励类型: {', '.join(irr_types)}")
    print(f"  子进程输出: {'显示' if show_main_output else '隐藏(更快)'}")
    print(f"  断点续跑: {'开启' if resume else '关闭'}")
    print("=" * 70 + "\n")
    
    # 生成训练集
    print(f"第一步: 生成训练集 ({num_train} 个仿真)")
    print("-" * 70)
    _generate_batch(
        num_sims=num_train,
        vehicle_types=vehicle_types,
        irr_types=irr_types,
        mileage_range=mileage_range,
        output_project=train_project,
        dataset_type="train",
        tz=tz,
        timeout=timeout,
        show_main_output=show_main_output,
        resume=resume,
    )
    
    # 生成测试集
    print(f"\n第二步: 生成测试集 ({num_test} 个仿真)")
    print("-" * 70)
    _generate_batch(
        num_sims=num_test,
        vehicle_types=vehicle_types,
        irr_types=irr_types,
        mileage_range=mileage_range,
        output_project=test_project,
        dataset_type="test",
        tz=tz,
        timeout=timeout,
        show_main_output=show_main_output,
        resume=resume,
    )
    
    print("\n" + "=" * 70)
    print("✓ 数据生成完成！")
    print("=" * 70)
    print(f"训练集路径: results/{train_project}/")
    print(f"测试集路径: results/{test_project}/")
    print("下一步: 运行训练脚本")
    print(f"  python PINO/VTCM_physicis_informed_fno.py")
    print("=" * 70 + "\n")


def _generate_batch(
    num_sims: int,
    vehicle_types: list,
    irr_types: list,
    mileage_range: tuple,
    output_project: str,
    dataset_type: str = "train",
    tz: float = 5.0,
    timeout: int = 12000,
    show_main_output: bool = True,
    resume: bool = True,
) -> None:
    """生成一批仿真"""
    
    configs = list(itertools.product(vehicle_types, irr_types))
    min_mileage, max_mileage = mileage_range
    
    success_count = 0
    fail_count = 0

    completed_before = _count_completed_runs(output_project) if resume else 0
    start_idx = min(completed_before, num_sims)

    if resume:
        print(f"  [恢复模式] 项目 {output_project} 已存在 {completed_before} 个完成样本，目标 {num_sims} 个")
    if start_idx >= num_sims:
        print(f"  [跳过] {dataset_type} 数据已满足目标数量，无需继续生成。")
        return
    
    for sim_idx in range(start_idx, num_sims):
        # 轮转配置
        config_idx = sim_idx % len(configs)
        vehicle_type, irr_type = configs[config_idx]
        
        # 线性插值里程
        start_mileage = min_mileage + (max_mileage - min_mileage) * (sim_idx / max(num_sims - 1, 1))
        
        # 进度条显示
        progress = (sim_idx + 1) / num_sims * 100
        bar_len = 30
        filled = int(bar_len * (sim_idx + 1) / num_sims)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"\n[{bar}] {progress:5.1f}% [{sim_idx + 1:3d}/{num_sims:3d}] 里程: {start_mileage:8.3f} km")
        print("→ " + "─" * 68)
        
        cmd = [
            "python",
            "generate_main.py",
            f"--vehicle_type={vehicle_type}",
            f"--irr_type={irr_type}",
            f"--start_mileage={start_mileage}",
            f"--tz={tz}",
            f"--project_name={output_project}",
            f"--save_dof_mode=vehicle",
            "--save_data=On",
            "--plot_figs=Off",
        ]
        
        try:
            if show_main_output:
                # 调试模式：显示 generate_main.py 的全部输出（会变慢）
                result = subprocess.run(
                    cmd,
                    cwd="/workspace/VTCM_PYTHON",
                    check=True,
                    timeout=timeout,
                )
            else:
                # 快速模式：仅显示本脚本进度，丢弃子进程日志
                result = subprocess.run(
                    cmd,
                    cwd="/workspace/VTCM_PYTHON",
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=timeout,
                )
            success_count += 1
            print("← " + "─" * 68 + " ✓")
        except subprocess.TimeoutExpired:
            print("← " + "─" * 68 + " ✗ (超时)")
            fail_count += 1
        except subprocess.CalledProcessError as e:
            print("← " + "─" * 68 + " ✗ (错误)")
            fail_count += 1
    
    # 完成统计
    print(f"\n  ╔{'═' * 66}╗")
    print(f"  ║ ✓ 成功: {success_count:3d}/{num_sims:3d}  |  ✗ 失败: {fail_count:3d}/{num_sims:3d}  {'':32s}║")
    print(f"  ╚{'═' * 66}╝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VTCM PINO 数据生成工具")
    parser.add_argument("--num_train", type=int, default=200, help="训练集仿真数量")
    parser.add_argument("--num_test", type=int, default=40, help="测试集仿真数量")
    parser.add_argument("--train_project", type=str, default="pino_train_dataset", help="训练集输出项目名")
    parser.add_argument("--test_project", type=str, default="pino_test_dataset", help="测试集输出项目名")
    parser.add_argument("--min_mileage", type=float, default=273.789599, help="最小起始里程 (km)")
    parser.add_argument("--max_mileage", type=float, default=273.789599, help="最大起始里程 (km)")
    parser.add_argument("--tz", type=float, default=5.0, help="单次仿真时长 (s)，越小越快")
    parser.add_argument("--timeout", type=int, default=12000, help="单次仿真超时 (s)")
    parser.add_argument("--hide_main_output", action="store_true", help="不显示 generate_main.py 输出（更快）") 
    parser.add_argument("--no_resume", action="store_true", help="关闭续跑，强制按目标数量重新尝试")
        


        
    args = parser.parse_args()
    
    generate_simulations(
        num_train=args.num_train,
        num_test=args.num_test,
        mileage_range=(args.min_mileage, args.max_mileage),
        train_project=args.train_project,
        test_project=args.test_project,
        tz=args.tz,
        timeout=args.timeout,
        show_main_output=not args.hide_main_output,
        resume=not args.no_resume,
    )
