'''
Author: 2099193635 2099193635@qq.com
Date: 2026-04-25 14:39:10
LastEditors: 2099193635 2099193635@qq.com
LastEditTime: 2026-04-29 10:34:41
FilePath: /VTCM_PYTHON/inverse_model/physics_informed.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import argparse
import os
import sys
from pathlib import Path

# 优先将项目根目录插入 sys.path，确保所有子包可被找到
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import torch
from physicsnemo.sym.eq.pde import PDE
from physics_modules.contact_geometry import WheelRailContactProcessor
from configs.parameters import *
from sympy import Symbol, Function, Number, Max, Rational

class VTCM(PDE):
    
	name = "VTCM"
	
	def __init__(self,
			vehicle_type: str,
			param_profile_dir: str,
			rail_type: str,
			faster_type: str,
			subrail_type: str,
			start_mileage: float,
			Lc: float,
			Lt: float,
			V: float,):
		# 直接初始化 equations 字典，绕开 PDE.super().__init__() 对 Variables() 的依赖
		self.equations = {}
		self.veh_emu = VehicleParams(vehicle_type = vehicle_type, yaml_dir = param_profile_dir)
		self.rail = RailParams(rail_type = rail_type, yaml_dir= param_profile_dir)
		self.Fastener_kv = Fastener_KV(fastener_type = faster_type, yaml_dir= param_profile_dir)
		self.subrail_standard = Subrail_Params(subrail_type = subrail_type, 	yaml_dir= param_profile_dir)
		self.mode_params = ModesParameters()
		i_v = Symbol("i_v")  
		t = Symbol("t")
		Z0 = Function("Z0")
		Zc = Function("Zc")(t)
		Zt1 = Function("Zt1")(t)
		Zt2 = Function("Zt2")(t)
		Zw1 = Function("Zw1")(t)
		Zw2 = Function("Zw2")(t)
		Zw3 = Function("Zw3")(t)
		Zw4 = Function("Zw4")(t)
		dt1 = - (Lc + Lt ) / V
		dt2 = - Lc / V
		dt3 = -dt2
		dt4 = -dt1
		Z0_w1 = Z0(t)
		Z0_w2 = Z0(t - dt2)
		Z0_w3 = Z0(t - dt3)
		Z0_w4 = Z0(t - dt4)
		processor = WheelRailContactProcessor()
		# rail_raw = np.loadtxt('Profile_file/rail_fade.txt')
		# wheel_raw = np.loadtxt('Profile_file/wheel_fade.txt') 
		# geom_info = processor.process_pre_information(rail_raw, wheel_raw)
		Ksz = self.veh_emu.Ksz
		Csz = self.veh_emu.Csz
		Kpz = self.veh_emu.Kpz
		Cpz = self.veh_emu.Cpz
		Mw = self.veh_emu.Mw
		F_sz1 = Ksz * (Zc - Zt1) + Csz * (Zc.diff(t) - Zt1.diff(t))
		F_sz2 = Ksz * (Zc - Zt2) + Csz * (Zc.diff(t) - Zt2.diff(t))

		F_pz1 = Kpz * (Zt1 - Zw1) + Cpz * (Zt1.diff(t) - Zw1.diff(t))
		F_pz2 = Kpz * (Zt1 - Zw2) + Cpz * (Zt1.diff(t) - Zw2.diff(t))
		F_pz3 = Kpz * (Zt2 - Zw3) + Cpz * (Zt2.diff(t) - Zw3.diff(t))
		F_pz4 = Kpz * (Zt2 - Zw4) + Cpz * (Zt2.diff(t) - Zw4.diff(t))

		P_kin1 = F_pz1 - 0.5 * Mw * Zw1.diff(t, 2)
		P_kin2 = F_pz2 - 0.5 * Mw * Zw2.diff(t, 2)
		P_kin3 = F_pz3 - 0.5 * Mw * Zw3.diff(t, 2)
		P_kin4 = F_pz4 - 0.5 * Mw * Zw4.diff(t, 2)
		Ghertz=4.5e-8
		delta_Z_kin1 = Ghertz * Max(P_kin1, 0)**(2/3)
		delta_Z_kin2 = Ghertz * Max(P_kin2, 0)**(2/3)
		delta_Z_kin3 = Ghertz * Max(P_kin3, 0)**(2/3)
		delta_Z_kin4 = Ghertz * Max(P_kin4, 0)**(2/3)

		delta_Z_geom1 = Zw1 - Z0_w1
		delta_Z_geom2 = Zw2 - Z0_w2
		delta_Z_geom3 = Zw3 - Z0_w3
		delta_Z_geom4 = Zw4 - Z0_w4
		Mc = self.veh_emu.Mc
		Mt = self.veh_emu.Mt
		acc_c_meas = Function("acc_c_meas")(t)

		self.equations = {}
		self.equations["car_body_constraint"] = Mc * acc_c_meas + 2 * F_sz1 + 2 * F_sz2

		self.equations["bogie_front"] = Mt * Zt1.diff(t, 2) - 2 * F_sz1 + 2 * F_pz1 + 2 * F_pz2
		self.equations["bogie_rear"] = Mt * Zt2.diff(t, 2) - 2 * F_sz2 + 2 * F_pz3 + 2 * F_pz4
		self.equations["nexus_w1"] = delta_Z_kin1 - delta_Z_geom1
		self.equations["nexus_w2"] = delta_Z_kin2 - delta_Z_geom2
		self.equations["nexus_w3"] = delta_Z_kin3 - delta_Z_geom3
		self.equations["nexus_w4"] = delta_Z_kin4 - delta_Z_geom4


class VTCMFull(PDE):
	"""
	VTCM PDE 改版 —— 兼容 PhysicsInformer。

	改造要点：
	  1. 所有状态量改用 Symbol（原始 VTCM 使用 Function(t)，PhysicsInformer 不支持对 t
	     自动求导）。
	  2. 速度 / 加速度作为独立 Symbol 显式传入（命名约定：Zc_v / Zc_a, Zt1_v / Zt1_a …）。
	  3. 时延轨道不平顺 Z0_w1..Z0_w4 作为外部 Symbol 输入（forward 时由调用方用
	     _time_shift_1d 预计算后传入）。
	  4. 2/3 次幂 (p23) 和 0.5*Mw (half_Mw) 作为 Symbol 常数（PhysicsInformer graph 中
	     会被当作标量张量传入）。

	使用方式：
	  vtcm_full = VTCMFull(Mc=..., Mt=..., Mw=..., Ksz=..., Csz=..., Kpz=..., Cpz=...)
	  pi = PhysicsInformer(
	      required_outputs=["car_body", "bogie_f", "bogie_r",
	                        "nexus_w1", "nexus_w2", "nexus_w3", "nexus_w4"],
	      equations=vtcm_full,
	      grad_method="finite_difference",
	      fd_dx=1e-4,
	      device="cuda",
	  )
	  residuals = pi.forward({
	      "Zc": ...,  "Zc_v": ..., "Zc_a": ...,  # [B, T]
	      ...
	      "Z0_w1": ..., "Z0_w2": ..., "Z0_w3": ..., "Z0_w4": ...,
	      "half_Mw": torch.tensor(0.5*Mw), "p23": torch.tensor(2/3),
	  })
	"""

	name = "VTCMFull"
	dim = 1

	def __init__(
		self,
		Mc: float,
		Mt: float,
		Mw: float,
		Ksz: float,
		Csz: float,
		Kpz: float,
		Cpz: float,
		G: float = 4.5e-8,
		F0: float = 0.0,
		dynamic_G: bool = False,
	):
		"""
		F0: 每轮对的静态轮轨力（N）。数据存储的是偏静平衡的动态偏差量，
		    正确的 Hertz 公式应使用绝对力：
		      nexus = G*(F0+P_kin)^(2/3) - G*F0^(2/3) = Zw - Z0
		    当 F0=0（默认）退化为旧行为：G*max(P_kin,0)^(2/3) = Zw - Z0。
		    推荐值：F0 = (Mc*g/4 + Mt*g/2 + Mw*g)，约 111 kN（高速客车）。

		dynamic_G: 若为 True，G 改为每轮对独立的 Symbol（G_w1..G_w4），
		    由调用方通过 VTCMFull.compute_Gwr_eff() 反算后传入 inputs dict。
		    此时构造参数 G 仅作为 compute_Gwr_eff 的缺省回退值，不影响方程。
		"""
		self._G_default = G
		self._F0 = F0
		self._Kpz = Kpz
		self._Cpz = Cpz
		self._Mw  = Mw
		self._dynamic_G = dynamic_G
		self.equations = {}
		self.dim = 1

		# ── 位移 Symbol（当前时刻）────────────────────────────────────────
		Zc  = Symbol("Zc");  Zt1 = Symbol("Zt1");  Zt2 = Symbol("Zt2")
		Zw1 = Symbol("Zw1"); Zw2 = Symbol("Zw2"); Zw3 = Symbol("Zw3"); Zw4 = Symbol("Zw4")

		# ── 速度 Symbol（由调用方 _first_derivative 预计算后传入）──────────
		Zc_v  = Symbol("Zc_v");  Zt1_v = Symbol("Zt1_v"); Zt2_v = Symbol("Zt2_v")
		Zw1_v = Symbol("Zw1_v"); Zw2_v = Symbol("Zw2_v")
		Zw3_v = Symbol("Zw3_v"); Zw4_v = Symbol("Zw4_v")

		# ── 加速度 Symbol（由调用方 _second_derivative 预计算后传入）────────
		Zc_a  = Symbol("Zc_a");  Zt1_a = Symbol("Zt1_a"); Zt2_a = Symbol("Zt2_a")
		Zw1_a = Symbol("Zw1_a"); Zw2_a = Symbol("Zw2_a")
		Zw3_a = Symbol("Zw3_a"); Zw4_a = Symbol("Zw4_a")

		# ── 时延轨道不平顺 Symbol（由调用方 _time_shift_1d 预计算后传入）────
		Z0_w1 = Symbol("Z0_w1"); Z0_w2 = Symbol("Z0_w2")
		Z0_w3 = Symbol("Z0_w3"); Z0_w4 = Symbol("Z0_w4")

		# ── 轨道不平顺加速度 Symbol（惯性修正项，由调用方有限差分得到）────────
		# 物理含义：在随轨道面运动的参考系中，接触力 F_contact 满足：
		#   F_contact = F_pz - hMw*(Zw_a - Z0_a)  = P_kin + hMw*Z0_a
		# 即轨道面加速时，绝对惯性力需减去轨道加速度贡献
		Z0_w1_a = Symbol("Z0_w1_a"); Z0_w2_a = Symbol("Z0_w2_a")
		Z0_w3_a = Symbol("Z0_w3_a"); Z0_w4_a = Symbol("Z0_w4_a")

		# ── 精确有理数幂次 & 内联半轮对质量（避免 float32 截断）──────────────
		p23  = Rational(2, 3)   # 精确 2/3，符号级精确，无 float 截断
		hMw  = Mw / 2           # Python float64 常数，精度优于外部 float32 Symbol

		# ── 静态 Hertz 压缩量（用于抵消静载偏置）──────────────────────────
		# 当 F0>0 时：nexus = G*(F0+P_kin)^(2/3) - delta0 = Zw - Z0
		# 当 F0=0 时：nexus = G*max(P_kin,0)^(2/3) = Zw - Z0（向后兼容）
		delta0 = float(G) * float(F0) ** float(Rational(2, 3)) if F0 > 0.0 else 0.0

		# ── 弹簧/阻尼力 ────────────────────────────────────────────────────
		F_sz1 = Ksz * (Zc  - Zt1) + Csz * (Zc_v  - Zt1_v)
		F_sz2 = Ksz * (Zc  - Zt2) + Csz * (Zc_v  - Zt2_v)
		F_pz1 = Kpz * (Zt1 - Zw1) + Cpz * (Zt1_v - Zw1_v)
		F_pz2 = Kpz * (Zt1 - Zw2) + Cpz * (Zt1_v - Zw2_v)
		F_pz3 = Kpz * (Zt2 - Zw3) + Cpz * (Zt2_v - Zw3_v)
		F_pz4 = Kpz * (Zt2 - Zw4) + Cpz * (Zt2_v - Zw4_v)

		# ── 轮轨法向动力项（含轨道加速度惯性修正） ──────────────────────────
		# P_i = F_pz_i - hMw*(Zw_i_a - Z0_wi_a)
		#     = F_pz_i - hMw*Zw_i_a + hMw*Z0_wi_a
		# 在轨道面参考系中，有效接触力由相对加速度决定，而非绝对加速度
		P1 = F_pz1 - hMw * Zw1_a + hMw * Z0_w1_a
		P2 = F_pz2 - hMw * Zw2_a + hMw * Z0_w2_a
		P3 = F_pz3 - hMw * Zw3_a + hMw * Z0_w3_a
		P4 = F_pz4 - hMw * Zw4_a + hMw * Z0_w4_a

		# ── nexus Hertz 接触公式 ─────────────────────────────────────────────
		# dynamic_G=False（默认）：使用固定 G float 或 F0 静载修正
		# dynamic_G=True         ：G 改为独立 Symbol，由 compute_Gwr_eff 提供
		if dynamic_G:
			# 每轮对独立 Hertz 常数 Symbol，由调用方在 inputs dict 中提供
			G_w1 = Symbol("G_w1"); G_w2 = Symbol("G_w2")
			G_w3 = Symbol("G_w3"); G_w4 = Symbol("G_w4")
			# 静态压缩量同样用各轮对 Symbol：delta0_wi = G_wi * F0^(2/3)
			if F0 > 0.0:
				F0_p23 = float(F0) ** float(Rational(2, 3))
				def _nexus(P, Gw):
					return Gw * Max(F0 + P, 0) ** p23 - Gw * F0_p23
			else:
				def _nexus(P, Gw):
					return Gw * Max(P, 0) ** p23
			G_vals = [G_w1, G_w2, G_w3, G_w4]
		else:
			if F0 > 0.0:
				def _nexus(P, Gw=G):
					return Gw * Max(F0 + P, 0) ** p23 - delta0
			else:
				def _nexus(P, Gw=G):
					return Gw * Max(P, 0) ** p23
			G_vals = [G, G, G, G]

		# ── 方程残差 ────────────────────────────────────────────────────────
		self.equations["car_body"] = Mc * Zc_a  + 2 * F_sz1 + 2 * F_sz2
		self.equations["bogie_f"]  = Mt * Zt1_a - 2 * F_sz1 + 2 * F_pz1 + 2 * F_pz2
		self.equations["bogie_r"]  = Mt * Zt2_a - 2 * F_sz2 + 2 * F_pz3 + 2 * F_pz4
		self.equations["nexus_w1"] = _nexus(P1, G_vals[0]) - (Zw1 - Z0_w1)
		self.equations["nexus_w2"] = _nexus(P2, G_vals[1]) - (Zw2 - Z0_w2)
		self.equations["nexus_w3"] = _nexus(P3, G_vals[2]) - (Zw3 - Z0_w3)
		self.equations["nexus_w4"] = _nexus(P4, G_vals[3]) - (Zw4 - Z0_w4)

	# ── 静态方法：从 batch 状态量反算每样本每轮对的 Hertz 常数 ─────────────
	@staticmethod
	def compute_Gwr_eff(
		inputs: dict,
		F0: float,
		Kpz: float,
		Cpz: float,
		Mw: float,
		G_fallback: float = 4.5e-8,
		min_contact_force: float = 5000.0,
		per_timestep: bool = True,
	) -> dict:
		"""
		从 PhysicsInformer inputs dict 中反算每条样本、每个轮对的有效
		Hertz 接触常数 Gwr_eff，并将结果以 Symbol 键名 G_w1..G_w4 写回 dict。

		反算公式（由 Hertz 方程变形得到，无循环依赖）：
		  G_eff(t) = (Zw(t) - Z0(t)) / ((F0 + P_kin(t))^(2/3) - F0^(2/3))
		  其中 P_kin = Kpz*(Zt - Zw) + Cpz*(Zt_v - Zw_v) - 0.5*Mw*Zw_a

		注意：P_kin 完全由悬挂力学决定，与 G 无关，因此反算无需迭代。

		参数
		----
		inputs        : PhysicsInformer 的 inputs dict，形如 {"Zw1": [B,T], ...}
		F0            : 静态轮轨力 (N)，每轮对均分的静载荷
		Kpz/Cpz/Mw   : 一系悬挂刚度/阻尼 (N/m, N·s/m) 及轮对质量 (kg)
		G_fallback    : 当接触力过小无法可靠估计时的回退値（若 per_sample_gfb=True 则用样本中位数替代）
		min_contact_force : 认为有效接触的最小绝对轮轨力阈値 (N)
		per_timestep  : True（默认）→ 逐时刻独立估计 G_eff(t)，形状 [B, T]；
		                False         → 每条样本取有效时刻的中位数后广播，形状 [B, T]。
		per_sample_gfb: True → 每样本用有效 G_eff 中位数作为回退値（显著优于固定 G_hertz）

		返回
		----
		inputs dict（原地修改并返回），新增键 "G_w1".."G_w4"，
		每个值形状 [B, T]，与其他输入张量同设备/同数据类型。
		"""
		# 轮对在构架上的对应关系：w1/w2 → Zt1，w3/w4 → Zt2
		bogie_map  = [("Zt1", "Zt1_v"), ("Zt1", "Zt1_v"),
		              ("Zt2", "Zt2_v"), ("Zt2", "Zt2_v")]
		wheel_keys = [("Zw1", "Zw1_v", "Zw1_a", "Z0_w1", "Z0_w1_a"),
		              ("Zw2", "Zw2_v", "Zw2_a", "Z0_w2", "Z0_w2_a"),
		              ("Zw3", "Zw3_v", "Zw3_a", "Z0_w3", "Z0_w3_a"),
		              ("Zw4", "Zw4_v", "Zw4_a", "Z0_w4", "Z0_w4_a")]

		hMw     = 0.5 * Mw      # 用于 Symbol 方程中的 P_kin
		full_Mw = Mw            # 用于 G_eff 反算中的真实动学轮轨力
		F0_p23  = float(F0) ** (2.0 / 3.0)   # F0^(2/3)

		for i, ((zt_k, ztv_k), (zw_k, zwv_k, zwa_k, z0_k, z0a_k)) in enumerate(
				zip(bogie_map, wheel_keys)):

			Zt   = inputs[zt_k];   Zt_v = inputs[ztv_k]
			Zw   = inputs[zw_k];   Zw_v = inputs[zwv_k]
			Zw_a = inputs[zwa_k];  Z0   = inputs[z0_k]
			# 轨道加速度修正（若已提供则使用，否则视为零）
			Z0_a = inputs.get(z0a_k, None)

			# P_kin 反算 G_eff 时用全体轮对质量 Mw（方程内的 P_i Symbol 不变）
			# 物理依据： GF_Wheelset_Z = -FNz + Fpz + Mw*g, 故 FNz = Fpz - Mw*Zw_a + Mw*g
			F_pz  = Kpz * (Zt - Zw) + Cpz * (Zt_v - Zw_v)
			P_kin = F_pz - full_Mw * Zw_a      # [B, T] 全轮对质量
			if Z0_a is not None:
				P_kin = P_kin + full_Mw * Z0_a  # 加入轨道加速度惯性修正

			# 绝对轮轨力（含静载）
			F_abs = F0 + P_kin                  # [B, T]

			# 分母：(F0+P)^(2/3) - F0^(2/3)
			denom = F_abs.clamp(min=0.0) ** (2.0 / 3.0) - F0_p23   # [B, T]

			# 分子：几何穿透量
			numer = Zw - Z0                     # [B, T]

			# 有效接触区掩码（|F_abs| 足够大且分母非零）
			valid = (F_abs.abs() > min_contact_force) & (denom.abs() > 1e-15)

			if per_timestep:
				# ── 逐时刻独立估计 G_eff(t) ────────────────────────────────
				# 无效时刻填充 G_fallback，避免除零
				safe_denom = denom.clone()
				safe_denom[~valid] = 1.0         # 临时填充，避免 NaN

				G_t = numer / safe_denom         # [B, T]
				G_t[~valid] = G_fallback         # 无效时刻回退到默认先设値

				# 使用每样本有效 G_eff 中位数作为回退値（显著优于固定 G_hertz）
				# 这消除了固定 fallback 带来的 DC 偏差它是当前时刻 G_eff 的最佳单数估计
				G_pos_valid = G_t[valid & (G_t > 0)]  # 有效且正候的 G_eff
				if G_pos_valid.numel() > 0:
					# 每样本独立的中位数
					G_fb_samples = torch.zeros(
						Zw.shape[0], dtype=Zw.dtype, device=Zw.device)
					for b in range(Zw.shape[0]):
						mask_b = valid[b] & (G_t[b] > 0)
						if mask_b.any():
							G_fb_samples[b] = G_t[b][mask_b].median()
						else:
							G_fb_samples[b] = G_fallback
					G_fb_per = G_fb_samples.unsqueeze(-1)  # [B, 1]
				else:
					G_fb_per = torch.full_like(Zw[:, :1], G_fallback)

				# 裁剪：允许幅度比 G_hertz 大得多的 G_eff（数据显示 G_eff~1e-6，截断上界掩10x）
				G_hi = (G_fb_per * 10.0).clamp(min=G_fallback)
				G_t = torch.where(
					(G_t > 0) & (G_t < G_hi),
					G_t,
					G_fb_per.expand_as(G_t)
				)
				inputs[f"G_w{i+1}"] = G_t       # [B, T]
			else:
				# ── 逐样本取中位数后广播（旧行为） ──────────────────────────
				G_eff_per_sample = torch.full(
					(Zw.shape[0],), G_fallback,
					dtype=Zw.dtype, device=Zw.device)

				for b in range(Zw.shape[0]):
					v = valid[b]
					if v.any():
						g_t = numer[b][v] / denom[b][v]
						G_eff_per_sample[b] = g_t.median()

				inputs[f"G_w{i+1}"] = G_eff_per_sample.unsqueeze(-1).expand_as(Zw)

		return inputs


if __name__ == "__main__":
    import h5py
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    HDF5_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'datasets', 'VTCM_inverse', 'train_full_seq.hdf5'
    )

    # ── 1. 读取第 0 条样本（直接使用 HDF5 中存储的物理状态量） ────────────
    with h5py.File(HDF5_PATH, 'r') as f:
        c_sample    = f['c'][0]              # (17,) 物理参数
        u_sample    = f['u'][0, :, 0]        # (T,) 输入激励（归一化不平顺）
        y_sample    = f['y'][0, :, 0]        # (T,) 输出响应（归一化车体加速度）
        vx_kmh      = float(f['vx'][0, 0])
        seq_len     = int(f['seq_lengths'][0])
        src_file    = f['source_file'][0]
        if isinstance(src_file, bytes):
            src_file = src_file.decode('utf-8')
        # 物理状态量（由 inverse_dataset_gen.py 写入，DOF顺序：Zc/Zt1/Zt2/Zw1/Zw2/Zw3/Zw4）
        phys_x  = f['phys_x'][0].astype(np.float64)   # (T, 7) 位移 m
        phys_v  = f['phys_v'][0].astype(np.float64)   # (T, 7) 速度 m/s
        phys_a  = f['phys_a'][0].astype(np.float64)   # (T, 7) 加速度 m/s²
        phys_z0 = f['phys_z0'][0, :, 0].astype(np.float64)  # (T,) 轮1处轨道不平顺 m

    vx_ms  = vx_kmh / 3.6
    Nt_eff = phys_x.shape[0]
    dt_sim = 1e-4   # 仿真步长 0.1ms（生成时固定）

    print("=" * 65)
    print(f"来源文件  : {src_file}")
    print(f"车速      : {vx_kmh:.2f} km/h  ({vx_ms:.4f} m/s)")
    print(f"序列长度  : {seq_len},  phys_x shape: {phys_x.shape}")

    # ── 2. 实例化 VTCM 物理方程类 ──────────────────────────────────────────
    print("\n正在实例化 VTCM PDE...")
    vtcm = VTCM(
        vehicle_type     = '高速客车',
        param_profile_dir= 'configs/standard',
        rail_type        = 'CHN60',
        faster_type      = 'Standard_KV',
        subrail_type     = 'Standard_Subrail',
        start_mileage    = 273.789599,
        Lc               = 9.0,
        Lt               = 1.2,
        V                = vx_ms,
    )
    print("VTCM 实例化成功！")
    print(f"  Mc={vtcm.veh_emu.Mc:.1f} kg, Mt={vtcm.veh_emu.Mt:.1f} kg, Mw={vtcm.veh_emu.Mw:.1f} kg")
    print(f"  Ksz={vtcm.veh_emu.Ksz:.3e} N/m, Csz={vtcm.veh_emu.Csz:.3e} N·s/m")
    print(f"  Kpz={vtcm.veh_emu.Kpz:.3e} N/m, Cpz={vtcm.veh_emu.Cpz:.3e} N·s/m")

    # ── 3. 从 phys_x/v/a 中拆分各 DOF（顺序与 physics_local_dofs 一致） ──
    # DOF 列映射: 0=Zc, 1=Zt1, 2=Zt2, 3=Zw1, 4=Zw2, 5=Zw3, 6=Zw4
    Zc_x,  Zc_v,  Zc_a  = phys_x[:,0], phys_v[:,0], phys_a[:,0]
    Zt1_x, Zt1_v, Zt1_a = phys_x[:,1], phys_v[:,1], phys_a[:,1]
    Zt2_x, Zt2_v, Zt2_a = phys_x[:,2], phys_v[:,2], phys_a[:,2]
    Zw1_x, Zw1_v, Zw1_a = phys_x[:,3], phys_v[:,3], phys_a[:,3]
    Zw2_x, Zw2_v, Zw2_a = phys_x[:,4], phys_v[:,4], phys_a[:,4]
    Zw3_x, Zw3_v, Zw3_a = phys_x[:,5], phys_v[:,5], phys_a[:,5]
    Zw4_x, Zw4_v, Zw4_a = phys_x[:,6], phys_v[:,6], phys_a[:,6]

    # ── 4. 保留 y/u 备用（此处仅验证物理残差，不做归一化对比） ──────────────
    # （若需要反归一化残差对比，可加载 datasets/VTCM_inverse/norm_stats.npz）

    # ── 5. 为各轮对构建 Z0（时延版本），phys_z0 是轮4处轨道不平顺（基准） ──────
    # 位置关系（来自 generate_main.py）：
    #   Xw1 = Xw4 + 2*(Lc+Lt), Xw2 = Xw4 + 2*Lc, Xw3 = Xw4 + 2*Lt
    # 轮1/2/3 先于轮4遇到不平顺，因此需向前查找（正步数 = 超前查找）
    Lc, Lt = 9.0, 1.2
    d1_exact = 2 * (Lc + Lt) / vx_ms / dt_sim   # 精确浮点延迟步数（轮1）
    d2_exact = 2 * Lc        / vx_ms / dt_sim   # 精确浮点延迟步数（轮2）
    d3_exact = 2 * Lt        / vx_ms / dt_sim   # 精确浮点延迟步数（轮3）
    print(f"\n精确延迟步数: w1={d1_exact:.4f}, w2={d2_exact:.4f}, w3={d3_exact:.4f}")
    print(f"舍入误差:     w1={d1_exact%1:.4f}步 ({(d1_exact%1)*vx_ms*dt_sim*1000:.3f}mm), "
          f"w2={d2_exact%1:.4f}步 ({(d2_exact%1)*vx_ms*dt_sim*1000:.3f}mm), "
          f"w3={d3_exact%1:.4f}步 ({(d3_exact%1)*vx_ms*dt_sim*1000:.3f}mm)")

    def time_shift_frac(arr, steps_float):
        """亚步长线性插值平移（超前查找：取未来 steps_float 步处的值）"""
        s_lo = int(steps_float)
        frac = steps_float - s_lo
        def _shift_int(a, s):
            if s == 0: return a.copy()
            out = np.empty_like(a)
            if s > 0:
                out[:s] = a[0]; out[s:] = a[:-s]
            else:
                out[s:] = a[-1]; out[:s] = a[s:]
            return out
        v_lo = _shift_int(arr, s_lo)
        if abs(frac) < 1e-9:
            return v_lo
        v_hi = _shift_int(arr, s_lo + 1)
        return (1.0 - frac) * v_lo + frac * v_hi

    Z0_w1 = time_shift_frac(phys_z0, d1_exact)   # 轮1：亚步长精确超前查找
    Z0_w2 = time_shift_frac(phys_z0, d2_exact)   # 轮2：亚步长精确超前查找
    Z0_w3 = time_shift_frac(phys_z0, d3_exact)   # 轮3：亚步长精确超前查找
    Z0_w4 = phys_z0.copy()                        # 轮4为基准，无需平移

    # ── 6. 计算弹簧/阻尼力及轮轨接触残差 ───────────────────────────────────
    Mc  = vtcm.veh_emu.Mc;  Mt  = vtcm.veh_emu.Mt;  Mw  = vtcm.veh_emu.Mw
    Ksz = vtcm.veh_emu.Ksz; Csz = vtcm.veh_emu.Csz
    Kpz = vtcm.veh_emu.Kpz; Cpz = vtcm.veh_emu.Cpz
    G   = 4.5e-8
    g_grav = 9.81
    # 静态轮轨力估算（每轮对）：车体均摊 + 构架均摊 + 轮对自重
    F0 = Mc * g_grav / 4 + Mt * g_grav / 2 + Mw * g_grav
    delta0 = G * F0 ** (2/3)   # 静态 Hertz 压缩量
    print(f"静态轮轨力 F0 = {F0:.0f} N ({F0/1000:.1f} kN), delta0 = {delta0*1e6:.2f} μm")

    F_sz1 = Ksz*(Zc_x - Zt1_x) + Csz*(Zc_v - Zt1_v)
    F_sz2 = Ksz*(Zc_x - Zt2_x) + Csz*(Zc_v - Zt2_v)
    F_pz1 = Kpz*(Zt1_x - Zw1_x) + Cpz*(Zt1_v - Zw1_v)
    F_pz2 = Kpz*(Zt1_x - Zw2_x) + Cpz*(Zt1_v - Zw2_v)
    F_pz3 = Kpz*(Zt2_x - Zw3_x) + Cpz*(Zt2_v - Zw3_v)
    F_pz4 = Kpz*(Zt2_x - Zw4_x) + Cpz*(Zt2_v - Zw4_v)

    P_kin = [F_pz1 - 0.5*Mw*Zw1_a,
             F_pz2 - 0.5*Mw*Zw2_a,
             F_pz3 - 0.5*Mw*Zw3_a,
             F_pz4 - 0.5*Mw*Zw4_a]
    Z0_ref  = [Z0_w1, Z0_w2, Z0_w3, Z0_w4]
    Zw_list = [Zw1_x, Zw2_x, Zw3_x, Zw4_x]
    dz_geom = [Zw_list[i] - Z0_ref[i] for i in range(4)]

    # 旧公式：G * max(P_kin, 0)^(2/3)
    dz_kin_old = [G * np.maximum(P, 0)**(2/3) for P in P_kin]
    # 新公式：G * (F0+P_kin)^(2/3) - delta0  ← 带静载修正，物理上更正确
    dz_kin_new = [G * np.maximum(F0 + P, 0)**(2/3) - delta0 for P in P_kin]

    # ── 7. 汇总残差 ─────────────────────────────────────────────────────────
    res_old = {}
    res_new = {}
    res_old["car_body"] = Mc * Zc_a + 2*F_sz1 + 2*F_sz2
    res_new["car_body"] = res_old["car_body"].copy()
    res_old["bogie_f"]  = Mt * Zt1_a - 2*F_sz1 + 2*F_pz1 + 2*F_pz2
    res_new["bogie_f"]  = res_old["bogie_f"].copy()
    res_old["bogie_r"]  = Mt * Zt2_a - 2*F_sz2 + 2*F_pz3 + 2*F_pz4
    res_new["bogie_r"]  = res_old["bogie_r"].copy()
    for i in range(4):
        res_old[f"nexus_w{i+1}"] = dz_kin_old[i] - dz_geom[i]
        res_new[f"nexus_w{i+1}"] = dz_kin_new[i] - dz_geom[i]

    print("\n=== 物理方程残差统计（全 HDF5 序列, Nt={} 步） ===".format(Nt_eff))
    print(f"  {'方程':<14}  {'旧公式 RMS':>14}  {'新公式 RMS':>14}  "
          f"{'改进量(μm)':>12}  {'Zw_std':>12}  {'旧/Zw_std':>10}  {'新/Zw_std':>10}")
    print("-" * 96)
    Zw_std_all = [Zw_list[i].std() if i < 4 else 1.0 for i in range(4)]
    for k in ["car_body", "bogie_f", "bogie_r",
               "nexus_w1", "nexus_w2", "nexus_w3", "nexus_w4"]:
        r_old = res_old[k]
        r_new = res_new[k]
        rms_o = np.sqrt(np.mean(r_old**2))
        rms_n = np.sqrt(np.mean(r_new**2))
        # nexus 的归一化尺度用对应 Zw_std
        if k.startswith("nexus_w"):
            i = int(k[-1]) - 1
            zw_s = Zw_std_all[i]
            improve = (rms_o - rms_n) * 1e6
            print(f"  {k:<14}  {rms_o:>14.4e}  {rms_n:>14.4e}  "
                  f"{improve:>12.2f}  {zw_s:>12.4e}  {rms_o/zw_s:>10.4f}  {rms_n/zw_s:>10.4f}")
        else:
            improve = (rms_o - rms_n) * 1e6
            print(f"  {k:<14}  {rms_o:>14.4e}  {rms_n:>14.4e}  {improve:>12.2f}")
    print("=" * 96)
    print("  注：新公式 = G*(F0+P_kin)^(2/3) - delta0，旧公式 = G*max(P_kin,0)^(2/3)")
    print(f"      延迟：亚步长精确插值（不再四舍五入到整数步）")

