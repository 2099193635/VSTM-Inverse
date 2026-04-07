from configs.parameters import VehicleParams, Antiyawer_parameters, ExtraForceElements_parameters
from physics_modules.suspension import SuspensionSystem
# 假设 RailParams, IntegrationParams 已经在您的 configs 中就绪
# 如果目前只要测试 VehicleParams，可以先注释掉另外两个

def test_data_structures():
    print("================ 测试 1：高速客车 (基础与加噪) ================")
    # 1. 默认初始化 (高速客车)
    veh_emu = VehicleParams(vehicle_type='高速客车')
    # suspension = SuspensionSystem(veh_emu)
    # (如果 sim 和 rail 尚未写好，可以先注释掉这两行)
    # rail = RailParams()
    # sim = IntegrationParams()
    # sim.setup(veh_emu)

    print(f"当前状态：无噪声基础模型 [{veh_emu.vehicle_type}]")
    print(f"车体基础质量 Mc: {veh_emu.Mc} kg")
    print(f"静轮重 P0: {veh_emu.P0:.2f} N")
    
    print("\n--- 正在为数据集生成加噪 ---")
    veh_emu.randomize_for_dataset(noise_ratio=0.15)
    # sim.setup(veh_emu, random_speed=True)
    
    print(f"当前状态：加噪扰动后 (噪声比例 ±15%)")
    print(f"加噪车体质量 Mc: {veh_emu.Mc:.2f} kg")
    print(f"新静轮重 P0: {veh_emu.P0:.2f} N")


    print("\n================ 测试 2：多车型拓扑切换 (重载货车) ================")
    # 2. 切换为重载货车进行测试
    veh_freight = VehicleParams(vehicle_type='普通货车_重车')
    
    print(f"当前状态：无噪声基础模型 [{veh_freight.vehicle_type}]")
    print(f"车体基础质量 Mc: {veh_freight.Mc} kg")
    print(f"货车独有摇枕质量 MB: {veh_freight.MB} kg")
    # 验证底层的 P0 计算是否成功应用了货车的 1摇枕+2侧架 公式
    print(f"静轮重 P0: {veh_freight.P0:.2f} N")

    print("\n================ 测试 3：抗蛇行减振器 (高速客车) ================")
    # 3. 验证抗蛇行减振器矩阵切片
    antiyaw = Antiyawer_parameters()
    print(f"抗蛇行减振器刚度 Kantiyawer: {antiyaw.kantiyawer:.1e} N/m")
    print(f"横向跨距之半 dsc: {antiyaw.dsc} m")
    print(f"提取的非线性阻尼速度节点 (yaw_damper_v): \n{antiyaw.yaw_damper_v}")
    print(f"提取的非线性阻尼力节点 (yaw_damper_f): \n{antiyaw.yaw_damper_f}")

    print("\n================ 测试 4：额外力元参数 (高速客车) ================")
    # 4. 验证依赖 Lc 的额外力元动态计算
    veh_emu_base = VehicleParams(vehicle_type='高速客车')
    print(f"获取当前车型 [{veh_emu_base.vehicle_type}] 的半定距 Lc: {veh_emu_base.Lc} m")
    
    # 将 Lc 喂给 ExtraForceElements_parameters
    extra_forces = ExtraForceElements_parameters(Lc=veh_emu_base.Lc)
    
    print("\n--- 节点与减振器基础参数 ---")
    print(f"一系垂向减振器阻尼 Cz_pvd: {extra_forces.Cz_pvd} N.s/m")
    print(f"转臂节点纵向刚度 Knodex: {extra_forces.Knodex:.1e} N/m")
    
    print("\n--- 动态位置与插值矩阵计算验证 ---")
    print(f"左侧 SLD 纵向位置数组 Lsld_Car_L: {extra_forces.Lsld_Car_L}")
    print(f"右侧 SLD 纵向位置数组 Lsld_Car_R: {extra_forces.Lsld_Car_R}")
    print(f"二系横向减振器(SLD)速度节点: {extra_forces.lat_damper_v}")
    print(f"二系横向减振器(SLD)阻尼力节点: {extra_forces.lat_damper_f}")

if __name__ == "__main__":
    test_data_structures()