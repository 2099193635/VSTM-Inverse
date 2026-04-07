"""VTCM coupled dynamics equations for PhysicsNeMo.

参考 physicsnemo.sym.eq.pdes.diffusion 的组织方式，
这里给出一个可直接用于 PINO / PINN 约束的“集成方程”类：

1) 运动学约束:
   x_t = v,
   v_t = a
2) 三体耦合动力学约束（车体 + 前后构架）:
   m a + coupling_forces - external_forces = 0

该类将所有约束写入 self.equations，便于直接构图为 PhysicsNeMo 节点。
"""

from sympy import Function, Number, Symbol

from physicsnemo.sym.eq.pde import PDE


class VTCMCoupledDynamics(PDE):
	"""
	VTCM Coupled Dynamics Equations
	"""

	name = "VTCMCoupledDynamics"

	def __init__(
		self,
		m_c='m_c',
		m_f='m_f',
		m_r='m_r',
		k_pz='k_pz',
		c_pz='c_pz',
		k_sz='k_sz',
		c_sz='c_sz',
		g='g',
		lc = 'lc',
		time=True,
	):
		self.time = time

		# coordinates / time
		t = Symbol("t")
		x = Symbol("x")
		# input variables
		input_variables = {"x": x, "t": t}
		if not self.time:
			input_variables.pop("t")

		# state variables
		x_c = Function("x_c")(*input_variables)
		v_c = Function("v_c")(*input_variables)
		a_c = Function("a_c")(*input_variables)
		roll_c = Function("roll_c")(*input_variables)  # 车体滚转角
		pitch_c = Function("pitch_c")(*input_variables)  # 车体俯仰角
		v_pitch_c = Function("v_pitch_c")(*input_variables)  # 车体俯仰角速度
		yaw_c = Function("yaw_c")(*input_variables)  # 车体偏航角

		x_f = Function("x_f")(*input_variables)
		v_f = Function("v_f")(*input_variables)
		a_f = Function("a_f")(*input_variables)
		roll_f = Function("roll_f")(*input_variables)  # 前构架滚转角
		pitch_f = Function("pitch_f")(*input_variables)  # 前构架俯仰角
		yaw_f = Function("yaw_f")(*input_variables)  # 前构架偏航角	

		x_r = Function("x_r")(*input_variables)
		v_r = Function("v_r")(*input_variables)
		a_r = Function("a_r")(*input_variables)
		roll_r = Function("roll_r")(*input_variables)  # 后构架滚转角
		pitch_r = Function("pitch_r")(*input_variables)  # 后构架俯仰角
		yaw_r = Function("yaw_r")(*input_variables)  # 后构架偏航角
	
		x_w1 = Function("x_w1")(*input_variables)  # 前轮1轨道输入
		v_w1 = Function("v_w1")(*input_variables)  # 前轮1轨道输入速度
		a_w1 = Function("a_w1")(*input_variables)  # 前轮1轨道输入加速度
		roll_w1 = Function("roll_w1")(*input_variables)  # 前轮1轨道输入滚转角
		pitch_w1 = Function("pitch_w1")(*input_variables)  # 前轮1轨道输入俯仰角
		yaw_w1 = Function("yaw_w1")(*input_variables)  # 前轮1轨道输入偏航角
		x_w2 = Function("x_w2")(*input_variables)  # 前轮2轨道输入
		v_w2 = Function("v_w2")(*input_variables)  # 前轮2轨道输入速度
		a_w2 = Function("a_w2")(*input_variables)  # 前轮2轨道输入加速度
		roll_w2 = Function("roll_w2")(*input_variables)  # 前轮2轨道输入滚转角
		pitch_w2 = Function("pitch_w2")(*input_variables)  # 前轮2轨道输入俯仰角
		yaw_w2 = Function("yaw_w2")(*input_variables)  # 前轮2轨道输入偏航角

		x_w3 = Function("x_w3")(*input_variables)  # 后轮1轨道输入
		v_w3 = Function("v_w3")(*input_variables)  # 后轮1轨道输入速度
		a_w3 = Function("a_w3")(*input_variables)  # 后轮1轨道输入加速度
		roll_w3 = Function("roll_w3")(*input_variables)  # 后轮1轨道输入滚转角
		pitch_w3 = Function("pitch_w3")(*input_variables)  # 后轮1轨道输入俯仰角
		yaw_w3 = Function("yaw_w3")(*input_variables)  # 后轮1轨道输入偏航角
		x_w4 = Function("x_w4")(*input_variables)  # 后轮2轨道输入
		v_w4 = Function("v_w4")(*input_variables)  # 后轮2轨道输入速度
		a_w4 = Function("a_w4")(*input_variables)  # 后轮2轨道输入加速度
		roll_w4 = Function("roll_w4")(*input_variables)  # 后轮2轨道输入滚转角
		pitch_w4 = Function("pitch_w4")(*input_variables)  # 后轮2轨道输入俯仰角
		yaw_w4 = Function("yaw_w4")(*input_variables)  # 后轮2轨道输入偏航角
		# helper: scalar or function field
		def _as_param(val, name):
			if isinstance(val, str):
				return Function(val)(*input_variables)
			if isinstance(val, (float, int)):
				return Number(val)
			return val

		# mass of carbody and bogies
		m_c = _as_param(m_c, "m_c")  # carbody mass
		m_f = _as_param(m_f, "m_f")	 # front bogie mass
		m_r = _as_param(m_r, "m_r")	 # rear bogie mass

		# stiffness and damping coefficients
		k_pz = _as_param(k_pz, "k_pz")  # primary suspension stiffness
		c_pz = _as_param(c_pz, "c_pz")  # primary suspension damping
		k_sz = _as_param(k_sz, "k_sz")  # secondary suspension stiffness
		c_sz = _as_param(c_sz, "c_sz")  # secondary suspension damping
		g = _as_param(g, "g")
		lc = _as_param(lc, "lc")
		# equations dict
		self.equations = {}

		# dynamic constraints (residual = 0)
		# carbody: mc * diff2(Zc) + 2 * Csz * diff(Zc) + 2 * Ksz * Zc - Csz * diff(Zt1) - Ksz * Zt1 - Csz * diff(Zt2) - Ksz * Zt2 - Mc * g = 0
		self.equations["dynamic_carbody"] = (
			m_c * a_c
			+ (c_sz + c_sz) * v_c
			+ (k_sz + k_sz) * x_c
			- c_sz * v_f
			- k_sz * x_f
			- c_sz * v_r
			- k_sz * x_r
		)
 
		# front bogie: Mt * diff2(Zt1) + (2 * Cpz + Csz) * diff(Zt1) + (2 * Kpz + Ksz) * Zt1 - Csz * diff(Zc) - Ksz * Zc - Cpz * diff(Zwl) - Cpz * diff(Zw2) - Kpz * Zwl - Kpz * Zw2 + Csz * lc * diff(beta_c) + Ksz * lc * beta_c - Mt * g = 0
		self.equations["dynamic_front_bogie"] = (
			m_f * a_f
			+ (2 * c_pz + c_sz) * v_f
			+ (2 * k_pz + k_sz) * x_f
			- c_sz * v_c
			- k_sz * x_c
			- c_pz * v_w1
			- c_pz * v_w2
			- k_pz * x_w1
			- k_pz * x_w2
			+ c_sz * lc * v_pitch_c
			+ k_sz * lc * pitch_c
		)

		# rear bogie: Mt * diff2(Zt2) + (2 * Cpz + Csz) * diff(Zt2) + (2 * Kpz + Ksz) * Zt2 - Csz * diff(Zc) - Ksz * Zc - Cpz * diff(Zw3) - Cpz * diff(Zw4) - Kpz * Zw3 - Kpz * Zw4 - Csz * lc * diff(beta_c) - Ksz * lc * beta_c - Mt * g = 0
		self.equations["dynamic_rear_bogie"] = (
			m_r * a_r
			+ (2 * c_pz + c_sz) * v_r
			+ (2 * k_pz + k_sz) * x_r
			- c_sz * v_c
			- k_sz * x_c
			- c_pz * v_w3
			- c_pz * v_w4
			- k_pz * x_w3
			- k_pz * x_w4
			- c_sz * lc * v_pitch_c
			- k_sz * lc * pitch_c
		)


if __name__ == "__main__":
	diff_eq = VTCMCoupledDynamics()
	diff_eq.pprint()

