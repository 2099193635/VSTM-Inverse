'''
Author: Niscienc 60505912+2099193635@users.noreply.github.com
Date: 2026-03-07 16:08:44
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-03-24 07:24:49
FilePath: /VTCM_PYTHON/generate_main.py
Description: 
Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
from configs.parameters import *
from defect_injector.irregularity import Irregularity
import numpy as np
import argparse
from configs.topology import SystemTopology
from physics_modules.contact_geometry import WheelRailContactProcessor
from physics_modules.suspension import SuspensionSystem
from physics_modules.wheel_rail_contact import WheelRailInteraction
from physics_modules.equation_of_motion import GeneralForceAssembler
from physics_modules.solver import DynamicSolver, SystemDynamics
from utils.post_processing import ResultPlotter
import datetime
import os
import json
from physics_modules.rail_modal import RailModalDynamics
from physics_modules.substructure import SubstructureDynamics


def parse_arguments():
    """统一定义并解析所有的外部输入参数"""
    parser = argparse.ArgumentParser(description="车辆-轨道耦合动力学仿真参数配置")
    
    # 1. 宏观运行参量
    parser.add_argument('--vx_set', type=float, default=215.0, help='车辆运行速度 (km/h)')
    parser.add_argument('--tz', type=float, default=5.0, help='仿真总时长 (s)')
    parser.add_argument('--tstep', type=float, default=1e-4, help='积分步长 (s)')
    parser.add_argument('--start_mileage', type=float, default=273.789599, help='仿真起始绝对里程 (km)')
    parser.add_argument('--curve_file_dir', type=str, default='preprocessing/台账/处理后/curve_parameters.csv', help='曲线参数文件路径')
    parser.add_argument('--gradient_file_dir', type=str, default='preprocessing/台账/处理后/gradient_parameters.csv', help='坡度参数文件路径')
    parser.add_argument('--cache_file_dir', type=str, default='configs/track_cache.npz', help='轨道缓存文件路径')
    parser.add_argument('--force_rebuild', type=str, default='On', choices=['On', 'Off'], help='是否强制重建力元（默认On）')

    # 2. 物理拓扑配置
    parser.add_argument('--vehicle_type', type=str, default='高速客车', help='车辆类型')
    parser.add_argument('--rail_type', type=str, default='CHN60', help='钢轨类型')
    parser.add_argument('--fastener_type', type=str, default='Standard_KV', help='扣件类型')
    parser.add_argument('--param_profile_dir', type=str, default='configs/standard', help='参数配置目录（标准参数: configs/standard；试验参数: configs/试验名）')
    
    # 3. 轨道激扰控制
    parser.add_argument('--irr_type', type=str, default='外部导入', help='激扰类型 (随机不平顺/谐波不平顺/无不平顺/外部导入)')
    parser.add_argument('--irr_lead_time', type=float, default=2.0, help='不平顺前置无激励时长(s)，随机/外部导入建议设为2.0')
    parser.add_argument('--psd_type', type=str, default='高铁谱', help='功率谱类型 (高铁谱/干线谱/美国谱/德国低干扰谱)')
    parser.add_argument('--defect_switch', type=str, default='off', choices=['on', 'off'], help='是否开启局部病害')
    parser.add_argument('--input_path', type=str, default='', help='不平顺输入路径')
    parser.add_argument('--output_path', type=str, default='', help='不平顺输出路径')
        # 外部不平顺配置
    parser.add_argument('--external_mileage_mode', type=str, default='absolute', choices=['absolute', 'relative'], help='外部不平顺里程模式')
    parser.add_argument('--external_distance_unit', type=str, default='km', choices=['m', 'km'], help='外部不平顺里程单位')
    parser.add_argument('--Type2', type=str, default='空间谱', choices=['空间谱', '时间谱', '时间序列'], help='外部不平顺文件类型（空间谱/时间谱/时间序列）')
    _ext_base   = 'preprocessing/静检数据/呼局/20210416/处理后/静检上行20210416-271-278.merged.aligned.external'
    _ext_prefix = '静检上行20210416-271-278.merged.aligned'
    parser.add_argument('--external_files', type=str, nargs='*',
                        default=[
                            f'VL={_ext_base}/{_ext_prefix}_VL.txt',
                            f'VR={_ext_base}/{_ext_prefix}_VR.txt',
                            f'LL={_ext_base}/{_ext_prefix}_LL.txt',
                            f'LR={_ext_base}/{_ext_prefix}_LR.txt',
                        ],
                        help='外部不平顺文件路径，格式: KEY=VALUE，支持 VL/VR/LL/LR 四个通道')

    # 4. 下部结构
    parser.add_argument('--N_sub', type=int, default=2000, help='轨道下部结构离散单元数量')
    parser.add_argument('--X0', type=float, default=20.0, help='仿真初始状态位移 (m)')

    # 5. 积分参数
    parser.add_argument('--alpha', type=float, default=0.5, help='Newmark-beta 方法的 alpha 参数')
    parser.add_argument('--beta', type=float, default=0.25, help='Newmark-beta 方法的 beta 参数')
    parser.add_argument('--g', type=float, default=9.81, help='重力加速度 (m/s^2)')

    # 6. 力元控制开关
    parser.add_argument('--switch_curve_track', type=str, default='Off', choices=['On', 'Off'], help='是否开启线型引起的附加力')
    parser.add_argument('--switch_2point_contact', type=str, default='Off', choices=['On', 'Off'], help='是否开启两点接触模型')
    parser.add_argument('--switch_extra_force_element', type=str, default='Off', choices=['On', 'Off'], help='是否开启额外力元')
    parser.add_argument('--switch_pad_zone', type=str, default='Off', choices=['On', 'Off'], help='是否开启扣件区分')
    parser.add_argument('--switch_pad_partition', type=str, default='Off', choices=['On', 'Off'], help='是否开启扣件分区')
    parser.add_argument('--switch_railcant_unsymmetric', type=str, default='Off', choices=['On', 'Off'], help='是否开启轨道超高非对称')
    
    # 7. 锁定控制开关
    parser.add_argument('--switch_lock_veh_non_z', type=str, default='On', choices=['On', 'Off'], help='是否锁定车辆非垂向自由度')
    parser.add_argument('--switch_lock_axlebox', type=str, default='On', choices=['On', 'Off'], help='是否锁定轴箱自由度')
    parser.add_argument('--switch_lock_substructure', type=str, default='Off', choices=['On', 'Off'], help='是否锁定轨道下部结构自由度')
    
    # 8. 输出与可视化控制
    parser.add_argument('--save_data', type=str, default='On', choices=['On', 'Off'], help='是否将结果保存到本地')
    parser.add_argument('--save_dof_mode', type=str, default='vehicle', choices=['full', 'vehicle'], help='结果保存自由度模式：full=完整系统，vehicle=仅车体自由度')
    parser.add_argument('--project_name', type=str, default='default_project', help='结果项目名，保存路径为 results/项目名/运行名')
    parser.add_argument('--run_note', type=str, default='', help='试验描述（会写入结果目录名与参数归档）')
    parser.add_argument('--plot_figs', type=str, default='On', choices=['On', 'Off'], help='仿真完成后是否自动弹出图表')
    # (如果在 Jupyter/IDE 中直接运行，用 parse_known_args 防止参数识别报错)
    args, _ = parser.parse_known_args()
    return args
    

def main(args):
    #=========================Part1 导入运算参数=========================#
    # 1. 车辆系统参数
    veh_emu = VehicleParams(vehicle_type=args.vehicle_type, yaml_dir=args.param_profile_dir)
    # 2. 钢轨参数
    rail = RailParams(rail_type=args.rail_type, yaml_dir=args.param_profile_dir)
    # 3. 扣件参数
    faster_kv = Fastener_KV(fastener_type=args.fastener_type, yaml_dir=args.param_profile_dir)
    fdkv_params = FastenerFDKVParams(temperature=20, fdkv_switch='ON', yaml_dir=args.param_profile_dir)
    # 4. 轨下结构参数
    subrail_standard = Subrail_Params(subrail_type='Standard_Subrail', yaml_dir=args.param_profile_dir)
    print(f" -> [参数配置] 已加载参数目录: {args.param_profile_dir}")
    track_alignment = RealTrackAlignment(
        curve_file_dir= args.curve_file_dir,
        gradient_file_dir=args.gradient_file_dir,
        cache_file_dir=args.cache_file_dir,
        force_rebuild = args.force_rebuild
    )
    start_mileage_m = args.start_mileage * 1000.0
    test_s_straight = start_mileage_m
    # 5. 线型参数
    k1, h1, g1, Rcurve, Thetacurve, Lcurve, curvecase, L1, L2, Lz1, Lz2, hcg, S, ZH_abs = track_alignment.get_geometry_at(test_s_straight)

    #=========================Part2 生成循环积分参数、组件=========================#
    # 1. 设置钢轨模态求解数量
    mode_params = ModesParameters()
    # 2. 设置数值积分参数
    S0_start = start_mileage_m
    _irr_type_norm = str(args.irr_type).strip()
    _need_irre_lead = _irr_type_norm in ('随机不平顺', '外部导入')
    _tz_effective = float(args.tz) + (float(args.irr_lead_time) if _need_irre_lead else 0.0)
    if _need_irre_lead and float(args.irr_lead_time) > 0:
        print(f" -> [不平顺前置缓冲] 已启用 {float(args.irr_lead_time):.2f}s 无不平顺工况，总仿真时长: {_tz_effective:.2f}s")

    integration = IntegrationParams(
        Lc=veh_emu.Lc,                      # 直接从车辆实例导入
        Lt=veh_emu.Lt,                      # 直接从车辆实例导入
        R=veh_emu.R,                        # 直接从车辆实例导入
        Lkj=faster_kv.Lkj,                  # 直接从扣件实例导入
        Vx_set=args.vx_set,                 # 设定车速 300 km/h
        Tz=_tz_effective,                   # 随机/外部导入时自动叠加前置无不平顺时长
        Tstep=args.tstep,
        S0_mileage=S0_start,                # 设定地球绝对坐标起点
        Nsub=args.N_sub,                    # 设定轨下结构离散单元数量
        X0 = args.X0,                       # 设定仿真初始状态位移
        alpha = args.alpha,                 # Newmark-beta 方法的 alpha 参数
        beta = args.beta,                   # Newmark-beta 方法的 beta 参数
        g = args.g                         # 重力加速度

    )

    sim_switches = ExtraforceElementSwitch(
            Switch_CurveTrack=args.switch_curve_track,
            Switch_2PointContact=args.switch_2point_contact,
            Switch_ExtraForceElement = args.switch_extra_force_element,
            Switch_PadZone = args.switch_pad_zone,
            Switch_PadPartition = args.switch_pad_partition,
            Switch_RailCant_Unsymmetric = args.switch_railcant_unsymmetric
        )
    
    # === 预计算各计算步各轮对位置的线型参数 (曲率 / 超高 / 坡度 及其时间变化率) ===
    # 问题根源：原代码仅在 s0_start+1000 处采样一次，导致曲率/超高在整个积分中
    # 始终为常数，车辆通过不同线型时无法感知线型变化。
    # 修复：按每步各轮对的真实绝对里程预计算 (Nt, 4) 参数矩阵，供 solver 每步索引取用。
    _t_vec = np.arange(integration.Nt) * integration.Tstep
    _s4    = integration.S0_mileage + integration.X0 + integration.Vc * _t_vec   # 4位轮对绝对里程
    _Lc, _Lt = integration.Lc, integration.Lt
    # 7个关键位置的绝对里程 (Nt,7):
    #   0=Xw1, 1=Xw2, 2=Xw3, 3=Xw4(=X0t), 4=Xt1, 5=Xt2, 6=Xc
    # 对应 MATLAB Force_EquivalentCurveForce.m 中各构件中心位置计算：
    #   Xw4=X0t, Xw3=X0t+2*Lt, Xw2=X0t+2*Lc, Xw1=X0t+2*(Lt+Lc)
    #   Xt1=Xw1-Lt=X0t+2*Lc+Lt, Xt2=X0t+Lt, Xc=Xw3+Lc-Lt=X0t+Lc+Lt
    _all_s = np.column_stack([
        _s4 + 2*(_Lc + _Lt),    # col 0: Xw1 (1位轮对)
        _s4 + 2*_Lc,            # col 1: Xw2 (2位轮对)
        _s4 + 2*_Lt,            # col 2: Xw3 (3位轮对)
        _s4,                    # col 3: Xw4 (4位轮对 / X0t 基准)
        _s4 + 2*_Lc + _Lt,      # col 4: Xt1 (1号构架中心)
        _s4 + _Lt,              # col 5: Xt2 (2号构架中心)
        _s4 + _Lc + _Lt,        # col 6: Xc  (车体中心)
    ])  # shape: (Nt, 7)

    _sg = track_alignment.s_grid
    _k_mat  = np.column_stack([np.interp(_all_s[:, j], _sg, track_alignment.k_grid) for j in range(7)])
    _h_mat  = np.column_stack([np.interp(_all_s[:, j], _sg, track_alignment.h_grid) for j in range(7)])
    _g_mat  = np.column_stack([np.interp(_all_s[:, j], _sg, track_alignment.g_grid) for j in range(7)])
    # 时间变化率（中心差分；边界用单侧差分）
    _dk_mat  = np.gradient(_k_mat,  integration.Tstep, axis=0)   # d(K)/dt (Nt, 7)
    _dh_mat  = np.gradient(_h_mat,  integration.Tstep, axis=0)   # d(H)/dt = dTheta/dt (Nt, 7)
    _ddh_mat = np.gradient(_dh_mat, integration.Tstep, axis=0)   # d²(H)/dt² (Nt, 7)

    track_geometry = {
        'K'  : _k_mat,     # 曲率 1/m                     (Nt, 7)
        'H'  : _h_mat,     # 超高角 rad（无符号绝对值）   (Nt, 7)
        'G'  : _g_mat,     # 坡度（无量纲）               (Nt, 7)
        'dK' : _dk_mat,    # 曲率变化率 1/m/s             (Nt, 7)
        'dH' : _dh_mat,    # 超高角变化率 rad/s           (Nt, 7)
        'ddH': _ddh_mat,   # 超高角二阶变化率 rad/s²      (Nt, 7)
        'S'  : _all_s      # 7个位置的绝对里程 m          (Nt, 7)
    }
    print(f" -> [线型预计算] 完成! 曲率范围: [{_k_mat.min():.5f}, {_k_mat.max():.5f}] 1/m, "
          f"超高范围: [{_h_mat.min()*1000:.1f}, {_h_mat.max()*1000:.1f}] mm")

    # 兼容别名："时间序列" 统一映射为 irregularity 内部使用的 "时间谱"
    _type2 = '时间谱' if args.Type2 == '时间序列' else args.Type2

    track_simulator = Irregularity(
        Lc=integration.Lc, 
        Lt=integration.Lt, 
        Vc=integration.Vc,       # 直接读取中枢算好的 m/s 速度
        Tstep=integration.Tstep, # 直接读取中枢步长
        Tz=integration.Tz,       # 直接读取中枢(可能被防脱轨截断后)的真实时长
        Nt=integration.Nt,       # 直接读取中枢算好的总步数
        type=args.irr_type,
        Tstart=max(0.0, float(args.irr_lead_time)),  # 不平顺起步前无激励时长(s)
        Type2 = _type2,      # 时间谱/空间谱
        powerSpectrum_type=args.psd_type, 
        mile=2000,                # 里程池给足即可
        external_mileage_mode = args.external_mileage_mode,  # 外部里程模式（绝对里程/相对里程）
        external_distance_unit = args.external_distance_unit,  # 外部里程单位（m/km）
        external_start_mileage = args.start_mileage,   # 外部里程起点（绝对/相对里程均可，单位由 external_distance_unit 控制）
        input_path = args.input_path,       # 不平顺输入路径
        output_path = args.output_path      # 不平顺输出路径
    )
    # 将 KEY=VALUE 列表解析为字典，并自动注入 start_mileage
    _external_files_dict = {}
    for _item in (args.external_files or []):
        if '=' in _item:
            _k, _v = _item.split('=', 1)
            _external_files_dict[_k.strip()] = _v.strip()
    # 注意：仅在 absolute 模式下默认注入 start_mileage，避免 relative 文件被错误平移
    if args.external_mileage_mode == 'absolute':
        _external_files_dict.setdefault('start_mileage', args.start_mileage)

    track_excitation = track_simulator.excitation_irregularity(
        defect_switch=args.defect_switch,
        external_files=_external_files_dict)
    bz_L, by_L, dbz_L, dby_L, bz_R, by_R, dbz_R, dby_R, a, L = track_excitation
    print(f"左轨垂向不平顺范围: [{bz_L.min()*1000:.2f}, {bz_L.max()*1000:.2f}] mm")
    print(f"右轨垂向不平顺范围: [{bz_R.min()*1000:.2f}, {bz_R.max()*1000:.2f}] mm")
    #=========================Part 3 实例化物理引擎与系统拓扑=========================#
    # 1. 提取接触几何前处理信息
    processor = WheelRailContactProcessor()
    rail_raw = np.loadtxt('Profile_file/rail_fade.txt')
    wheel_raw = np.loadtxt('Profile_file/wheel_fade.txt') 
    geom_info = processor.process_pre_information(rail_raw, wheel_raw)
    
    # 2. 实例化系统拓扑
    rail_modal_sys = RailModalDynamics(rail_params=rail, integration_params=integration, mode_params=mode_params)
    substructure_sys = SubstructureDynamics(fastener_params=faster_kv, rail_params=rail, subrail_params=subrail_standard)
    topo = SystemTopology(Nt=integration.Nt, Nsub=IntegrationParams.Nsub, NV=mode_params.NV, NL=mode_params.NL, NT=mode_params.NT)
    ap = Antiyawer_parameters(yaml_dir=args.param_profile_dir)
    ep = ExtraForceElements_parameters(Lc=veh_emu.Lc, yaml_dir=args.param_profile_dir)
    suspension_sys = SuspensionSystem(veh_params=veh_emu, antiyawer_params=ap, extra_params=ep)
    wr_interaction = WheelRailInteraction(geom_info, veh_params=veh_emu)
    gf_assembler = GeneralForceAssembler(
        veh_params=veh_emu, integration_params=integration, rail_params=rail, 
        subrail_params=subrail_standard, mode_params=mode_params, anitiyawer_params=ap
    )
    sys_dynamics = SystemDynamics(veh_params=veh_emu, veh_int=integration, para_subrail=subrail_standard, control_mode= mode_params, para_extra_force=ep)
    physics_engines = {
        'suspension': suspension_sys, 'contact': wr_interaction, 'assembler': gf_assembler,
        'dynamics': sys_dynamics, 'rail_modal': rail_modal_sys, 'substructure': substructure_sys,
        'veh_params': veh_emu   # 车辆参数对象，供曲线等效力计算使用
    }
    
    #=========================Part 4 启动主积分器求解=========================#
    solver = DynamicSolver(topology=topo, 
                            integration_params=integration,
                            switch_lock_veh_non_z = args.switch_lock_veh_non_z,  
                            switch_lock_axlebox = args.switch_lock_axlebox, 
                            switch_lock_substructure = args.switch_lock_substructure)
    X, V, A, spy_data = solver.solve(
        track_excitation=track_excitation,
        geom_info=geom_info,
        engines=physics_engines,
        track_geometry=track_geometry,
        sim_switches=sim_switches
    )

    # =========================Part 4.1 追加后处理元数据=========================#
    # 保存不平顺（采用第4轮对基准轨道激励，长度 Nt+1）
    # 约定：bz/by 为位移(m)，dbz/dby 为变化率(m/s)
    try:
        spy_data['Irre_bz_L_ref'] = bz_L[3, :]
        spy_data['Irre_bz_R_ref'] = bz_R[3, :]
        spy_data['Irre_by_L_ref'] = by_L[3, :]
        spy_data['Irre_by_R_ref'] = by_R[3, :]
        spy_data['Irre_dbz_L_ref'] = dbz_L[3, :]
        spy_data['Irre_dbz_R_ref'] = dbz_R[3, :]
        spy_data['Irre_dby_L_ref'] = dby_L[3, :]
        spy_data['Irre_dby_R_ref'] = dby_R[3, :]

        # 对应空间坐标（相对里程）
        irre_n = bz_L.shape[1]
        spy_data['Irre_distance_m'] = np.arange(irre_n) * integration.Vc * integration.Tstep
    except Exception as e:
        print(f" -> [警告] 不平顺附加数据写入失败: {e}")

    # 保存平纵断面（直接使用已预计算的 track_geometry，以 4位轮对(基准)为参考轨迹）
    try:
        sim_s_abs = track_geometry['S'][:, 3]               # 4位轮对绝对里程 (Nt,)
        k_profile = track_geometry['K'][:, 3]               # 4位轮对曲率
        h_profile = track_geometry['H'][:, 3]               # 4位轮对超高角
        g_profile = track_geometry['G'][:, 3]               # 4位轮对坡度

        ds = integration.Vc * integration.Tstep
        z_profile = np.cumsum(g_profile) * ds

        spy_data['Track_abs_mileage_m']   = sim_s_abs
        spy_data['Track_rel_mileage_m']   = sim_s_abs - sim_s_abs[0]
        spy_data['Track_curvature_1pm']   = k_profile
        spy_data['Track_cant_m']          = h_profile
        spy_data['Track_gradient']        = g_profile
        spy_data['Track_vertical_profile_m'] = z_profile
        # 附加保存全部 4 个轮对的曲率/超高矩阵，便于后处理对比
        spy_data['Track_K_all_ws'] = track_geometry['K'][:, :4]   # (Nt, 4) 仅轮对位置
        spy_data['Track_H_all_ws'] = track_geometry['H'][:, :4]   # (Nt, 4) 仅轮对位置
    except Exception as e:
        print(f" -> [警告] 平纵断面附加数据写入失败: {e}")

    print("\n===================================================================")
    print("               仿真计算完成！数据已缓存至内存。               ")
    print("===================================================================")

    #=========================Part 5 结果保存与可视化=========================#
    # 1. 自动保存数据至 results/<project_name>/<run_name>/files/
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _note = str(args.run_note).strip()
    _note = _note if _note else 'standard'
    run_name = f"{args.vehicle_type}-{args.irr_type}-{args.save_dof_mode}-{_note}-{timestamp}"
    project_results_root = os.path.join('results', ResultPlotter._sanitize_name(args.project_name))
    saved_npz_path = ResultPlotter.save_data(
        run_name=run_name,
        X=X,
        V=V,
        A=A,
        spy_dict=spy_data,
        dt=integration.Tstep,
        idx_car_start=topo.idx_Car[0],
        idx_car_end=topo.idx_Car[1],
        save_dof_mode=args.save_dof_mode,
        results_root=project_results_root
    )

    # 2. 在同一目录保存 argparse 参数字典
    try:
        args_dict = vars(args)
        args_file = os.path.join(os.path.dirname(saved_npz_path), 'argparse_params.json')
        with open(args_file, 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=2)
        print(f" -> [参数归档] argparse 参数已保存至: {args_file}")
    except Exception as e:
        print(f" -> [警告] argparse 参数保存失败: {e}")

    # 3. 保存 run_meta.yaml（参数目录、试验描述、覆盖信息等）
    try:
        files_dir = os.path.dirname(saved_npz_path)
        run_meta_file = os.path.join(files_dir, 'run_meta.yaml')

        profile_abs = os.path.abspath(args.param_profile_dir)
        standard_abs = os.path.abspath(os.path.join('configs', 'standard'))
        yaml_names = [
            'vehicle_params.yaml',
            'rail_params.yaml',
            'fastener_kv.yaml',
            'subrail_params.yaml',
            'fastener_fdkv_params.yaml',
            'extra_force_elements.yaml',
            'antiyawer_params.yaml',
        ]

        with open(run_meta_file, 'w', encoding='utf-8') as f:
            f.write(f"run_name: {run_name}\n")
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"vehicle_type: {args.vehicle_type}\n")
            f.write(f"irr_type: {args.irr_type}\n")
            f.write(f"save_dof_mode: {args.save_dof_mode}\n")
            f.write(f"project_name: {args.project_name}\n")
            f.write(f"project_results_root: {os.path.abspath(project_results_root)}\n")
            f.write(f"run_note: {str(args.run_note).strip() or 'standard'}\n")
            f.write(f"param_profile_dir: {args.param_profile_dir}\n")
            f.write(f"param_profile_dir_abs: {profile_abs}\n")
            f.write(f"standard_profile_dir_abs: {standard_abs}\n")
            f.write(f"tz_input_s: {float(args.tz):.6f}\n")
            f.write(f"irr_lead_time_s: {float(args.irr_lead_time):.6f}\n")
            f.write(f"tz_effective_s: {float(_tz_effective):.6f}\n")
            f.write(f"result_npz: {saved_npz_path}\n")
            f.write(f"argparse_params: {args_file if 'args_file' in locals() else ''}\n")
            f.write("yaml_sources:\n")
            for name in yaml_names:
                p_profile = os.path.join(profile_abs, name)
                p_standard = os.path.join(standard_abs, name)
                profile_exists = os.path.exists(p_profile)
                standard_exists = os.path.exists(p_standard)
                source = 'profile' if profile_exists else ('standard' if standard_exists else 'missing')
                used_path = p_profile if profile_exists else (p_standard if standard_exists else '')
                f.write(f"  {name}:\n")
                f.write(f"    source: {source}\n")
                f.write(f"    used_path: {used_path}\n")

        print(f" -> [参数归档] 运行元数据已保存至: {run_meta_file}")
    except Exception as e:
        print(f" -> [警告] run_meta.yaml 保存失败: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)