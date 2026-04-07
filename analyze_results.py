r'''
Author: Niscienc 60505912+2099193635@users.noreply.github.com
Date: 2026-03-13 19:10:32
LastEditors: Niscienc 60505912+2099193635@users.noreply.github.com
LastEditTime: 2026-03-15 22:15:24
FilePath: \VTCM_PYTHON\analyze_results.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''

import numpy as np
import os
from utils.post_processing import ResultPlotter


def find_latest_result(results_dir='results'):
    """Compatibility fallback: return latest flat .npz directly under results/."""
    if not os.path.isdir(results_dir):
        return None

    npz_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.lower().endswith('.npz')
    ]
    if not npz_files:
        return None

    return max(npz_files, key=os.path.getmtime)


def find_latest_run_folder(results_dir='results'):
    """Return latest run folder that contains files/*.npz, supporting nested project folders."""
    if not os.path.isdir(results_dir):
        return None

    candidates = []
    for root, dirs, _ in os.walk(results_dir):
        if os.path.basename(root).lower() != 'files':
            continue
        npz_list = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith('.npz')
        ]
        if npz_list:
            latest_npz_time = max(os.path.getmtime(p) for p in npz_list)
            candidates.append((latest_npz_time, os.path.dirname(root)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_latest_result_in_run(run_dir):
    """Return latest .npz in run_dir/files/."""
    files_dir = os.path.join(run_dir, 'files')
    if not os.path.isdir(files_dir):
        return None

    npz_files = [
        os.path.join(files_dir, f)
        for f in os.listdir(files_dir)
        if f.lower().endswith('.npz')
    ]
    if not npz_files:
        return None
    return max(npz_files, key=os.path.getmtime)


def load_and_analyze(filepath, save_dir, show=False):
    """Load NPZ data and reconstruct the structure for plotting."""
    print(f" -> Loading result file: {filepath} ...")
    
    # 1. 加载 npz 压缩包 (它在 Python 中表现得像一个字典)
    data = np.load(filepath)
    
    # 2. 提取基础变量
    A = data['A']
    dt = float(data['dt'])
    idx_car_start = int(data['idx_car_start'])
    Nt = A.shape[0]  # 根据矩阵行数自动推导总步数 Nt
    
    # 3. 将其余变量重新组装回 spy_dict
    spy_dict = {}
    standard_keys = ['X', 'V', 'A', 'dt', 'idx_car_start']
    for key in data.files:
        if key not in standard_keys:
            spy_dict[key] = data[key]
            
    print(f" -> Data loaded successfully. Steps: {Nt}, time step: {dt}s. Preparing extended plots...")
    
    # 4. 调用画图模块
    saved_paths = ResultPlotter.plot_core_responses(
        Nt=Nt, 
        dt=dt, 
        A=A, 
        spy_dict=spy_dict, 
        idx_car_start=idx_car_start,
        save_dir=save_dir,
        show=show
    )
    return saved_paths

if __name__ == "__main__":
    # Optional: set one of the following manually.
    # 1) target_run_folder can be either folder name under results/ or absolute path.
    target_run_folder = ''
    # 2) target_file can be a direct .npz path (highest priority).
    target_file = 'results/pino_train_dataset/高速客车-随机不平顺-vehicle-standard-20260324_072013/files/simulation_result.npz'

    selected_file = None
    selected_run_dir = None

    if target_file:
        selected_file = target_file
        abs_target = os.path.abspath(target_file)
        selected_run_dir = os.path.dirname(os.path.dirname(abs_target)) if os.path.basename(os.path.dirname(abs_target)).lower() == 'files' else None
    else:
        if target_run_folder:
            selected_run_dir = target_run_folder if os.path.isabs(target_run_folder) else os.path.join('results', target_run_folder)
        else:
            selected_run_dir = find_latest_run_folder('results')

        if selected_run_dir and os.path.isdir(selected_run_dir):
            selected_file = find_latest_result_in_run(selected_run_dir)

    # Compatibility fallback for legacy flat results/*.npz
    if not selected_file:
        selected_file = find_latest_result('results')
        selected_run_dir = None

    if selected_file and os.path.exists(selected_file):
        if selected_run_dir and os.path.isdir(selected_run_dir):
            figures_dir = os.path.join(selected_run_dir, 'figures')
        else:
            figures_dir = os.path.join('results', 'figures_legacy')

        print(f" -> Selected result file: {selected_file}")
        print(f" -> Figures will be saved to: {figures_dir}")
        load_and_analyze(selected_file, save_dir=figures_dir, show=False)
    else:
        print("No valid result file found in 'results'. Please run generate_main.py first.")