# `ldc_pinns/train.py` 逐段源码讲解

## 1. 文档定位

本文件讲解的是：

- [physicsnemo/examples/cfd/ldc_pinns/train.py](physicsnemo/examples/cfd/ldc_pinns/train.py)

这是一个 **纯 PINN 风格** 示例：

- 不依赖真实标签数据集
- 通过几何采样点 + PDE 残差 + 边界条件损失训练

你可以把它和 [darcy_physics_informed_fno_逐段源码讲解.md](darcy_physics_informed_fno_逐段源码讲解.md) 对照看，前者是 PINO，这个是 PINN。

---

## 2. 这个脚本在做什么

一句话：

> 用 MLP 拟合二维方腔流（Lid Driven Cavity）中的 $u,v,p$，并用 Navier-Stokes 残差和边界条件约束训练。

网络映射是：

$$
(x,y) \rightarrow (u(x,y), v(x,y), p(x,y))
$$

总损失是：

$$
\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{BC}
$$

其中：

- $\mathcal{L}_{PDE}$：连续方程 + 动量方程残差
- $\mathcal{L}_{BC}$：no-slip 与顶壁速度边界

---

## 3. 模块导入的意义

脚本导入了这些核心组件：

- `FullyConnected`：PINN 网络本体
- `NavierStokes`：PDE 定义
- `PhysicsInformer`：根据网络输出算 PDE 残差
- `Rectangle` + `GeometryDatapipe`：几何和采样点生成
- `DistributedManager`：设备与分布式管理

你会看到导入里还有一些未使用项（如 `FNO`, `StaticCaptureTraining`, `MSELoss`），这是示例脚本常见的“保留导入”，不影响主逻辑。

---

## 4. Hydra 入口

入口定义：

```python
@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def ldc_trainer(cfg: DictConfig) -> None:
```

配置文件是：

- [physicsnemo/examples/cfd/ldc_pinns/config.yaml](physicsnemo/examples/cfd/ldc_pinns/config.yaml)

你真正用到的关键参数是：

- `scheduler.initial_lr`

当前值：

- `1e-3`

---

## 5. 初始化设备与日志

```python
DistributedManager.initialize()
dist = DistributedManager()

log = PythonLogger(name="ldc")
log.file_logging()
```

这里做了两件事：

1. 拿到统一设备句柄 `dist.device`（CPU/GPU）
2. 初始化文件日志

虽然是单卡也可运行，但这套写法兼容分布式环境。

---

## 6. 几何定义：方腔域

```python
height = 0.1
width = 0.1
rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))
```

定义的是二维矩形域：

$$
x,y \in [-0.05, 0.05]
$$

这就是 LDC 计算域。

---

## 7. 模型定义：经典 PINN 形式

```python
model = FullyConnected(
    in_features=2, out_features=3, num_layers=6, layer_size=512
).to(dist.device)
```

含义：

- 输入 2 维：`x,y`
- 输出 3 维：`u,v,p`
- 6 层隐层、每层 512

这就是最典型的 PINN MLP 结构。

---

## 8. PDE 与残差计算器

```python
ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
phy_inf = PhysicsInformer(
    required_outputs=["continuity", "momentum_x", "momentum_y"],
    equations=ns,
    grad_method="autodiff",
    device=dist.device,
)
```

关键信息：

- PDE：二维稳态 Navier-Stokes
- 需要残差项：连续方程、x 动量、y 动量
- 导数方法：`autodiff`

这也是它和 PINO 示例最关键的区别之一：

- PINN 常用 `autodiff`
- PINO 常用有限差分 / 频域差分

---

## 9. 优化器与学习率策略

```python
optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
scheduler = lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: 0.9999871767586216**step
)
```

学习率每个 step 乘一个常数，逐步衰减。

---

## 10. 推理可视化网格

```python
x = np.linspace(-0.05, 0.05, 512)
y = np.linspace(-0.05, 0.05, 512)
xx, yy = np.meshgrid(x, y, indexing="xy")
```

这不是训练点，而是每 1000 步做一次可视化时用的规则网格。

用于输出：

- `u` 场
- `v` 场
- `p` 场
- 速度模长

---

## 11. 数据不是标签集，而是几何采样点

### 11.1 边界点采样

```python
bc_dataloader = GeometryDatapipe(
    geom_objects=[rec],
    batch_size=1,
    num_points=2000,
    sample_type="surface",
    device=dist.device,
    num_workers=1,
    requested_vars=["x", "y"],
)
```

每次从边界采 2000 个点。

### 11.2 内部点采样

```python
interior_dataloader = GeometryDatapipe(
    geom_objects=[rec],
    batch_size=1,
    num_points=4000,
    sample_type="volume",
    device=dist.device,
    num_workers=1,
    requested_vars=["x", "y", "sdf"],
)
```

每次在体内采 4000 个点，并额外拿 `sdf`（signed distance function）。

这就是 PINN 的“训练数据来源”：几何采样，不是真值标签。

---

## 12. 主训练循环结构

```python
for i in range(10000):
    for bc_data, int_data in zip(bc_dataloader, interior_dataloader):
        ...
```

外层 10000 轮，内层每轮拿一批边界点和内部点。

---

## 13. 边界点拆分：no-slip 与 top wall

```python
y_vals = bc_data[0]["y"]
mask_no_slip = y_vals < height / 2
mask_top_wall = y_vals == height / 2
```

然后把边界点拆成两类：

- `no_slip`：除顶部外边界
- `top_wall`：顶部边界

其中顶部边界设定滑移速度（`u=1`），其余边界无滑移（`u=v=0`）。

---

## 14. 内部点 `requires_grad` 的关键设置

```python
for k, v in int_data[0].items():
    if k in ["x", "y"]:
        requires_grad = True
    else:
        requires_grad = False
    interior[k] = v.reshape(-1, 1).requires_grad_(requires_grad)
```

这是 PINN 自动微分的关键。

因为 PDE 里要计算对 `x,y` 的导数，所以坐标张量必须 `requires_grad=True`。

---

## 15. 前向预测

```python
coords = torch.cat([interior["x"], interior["y"]], dim=1)
no_slip_out = model(torch.cat([no_slip["x"], no_slip["y"]], dim=1))
top_wall_out = model(torch.cat([top_wall["x"], top_wall["y"]], dim=1))
interior_out = model(coords)
```

同一个网络分别在：

- no-slip 边界点
- top wall 点
- 内部点

上做前向。

---

## 16. 边界条件损失构造

```python
v_no_slip = torch.mean(no_slip_out[:, 1:2] ** 2)
u_no_slip = torch.mean(no_slip_out[:, 0:1] ** 2)
u_slip = torch.mean(
    ((top_wall_out[:, 0:1] - 1.0) ** 2)
    * (1 - 20 * torch.abs(top_wall["x"]))
)
v_slip = torch.mean(top_wall_out[:, 1:2] ** 2)
```

解释：

- `u_no_slip`, `v_no_slip`：无滑移边界约束
- `u_slip`：顶部速度逼近 1
- `v_slip`：顶部法向速度为 0

### 16.1 `u_slip` 的权重函数

`(1 - 20*|x|)` 让边缘权重更低（注释里写了 “weight the edges zero”），用于弱化顶角奇异区对训练的影响。

---

## 17. PDE 残差损失构造

```python
phy_loss_dict = phy_inf.forward(
    {
        "coordinates": coords,
        "u": interior_out[:, 0:1],
        "v": interior_out[:, 1:2],
        "p": interior_out[:, 2:3],
    }
)

cont = phy_loss_dict["continuity"] * interior["sdf"]
mom_x = phy_loss_dict["momentum_x"] * interior["sdf"]
mom_y = phy_loss_dict["momentum_y"] * interior["sdf"]
```

关键点：

1. `PhysicsInformer` 根据 Navier-Stokes 和自动微分输出残差
2. 残差被乘上 `sdf`

乘 `sdf` 的直觉是：

- 让靠近边界处的体内残差权重变小
- 增强训练稳定性（尤其边界附近导数更敏感时）

---

## 18. 总损失与更新

```python
phy_loss = (
    1 * torch.mean(cont**2)
    + 1 * torch.mean(mom_x**2)
    + 1 * torch.mean(mom_y**2)
    + u_no_slip
    + v_no_slip
    + u_slip
    + v_slip
)
phy_loss.backward()
optimizer.step()
scheduler.step()
```

本例中 PDE 三项残差都以同权重进入损失。

你后续做自己问题时，通常会需要调：

- PDE 各项权重
- BC 与 PDE 的相对权重

---

## 19. 周期性可视化输出

每 `1000` 步会：

1. 在规则网格上推理
2. 画 4 张图：`u`, `v`, `p`, `u_mag`
3. 保存到 `./outputs/outputs_pc_{i}.png`

并打印：

- 当前损失
- 当前学习率

这有助于判断是否收敛到合理流场。

---

## 20. 这个脚本体现的 PINN 关键模式

总结为 6 条：

1. **网络是坐标到场变量映射**：`(x,y)->(u,v,p)`
2. **训练点来自几何采样**，不是标签数据集
3. **PDE 残差通过自动微分计算**
4. **边界条件作为显式损失项加入**
5. **体内残差可用 `sdf` 重加权**
6. **可视化是训练可解释性的关键工具**

---

## 21. 与 `darcy_physics_informed_fno.py` 的对比

| 维度 | `ldc_pinns/train.py` | `darcy_physics_informed_fno.py` |
|---|---|---|
| 方法类型 | PINN | PINO |
| 网络 | MLP (`FullyConnected`) | FNO |
| 输入 | 坐标点 | 系数场网格 |
| 输出 | 点值 `u,v,p` | 整场 `u` |
| 导数 | `autodiff` | `finite_difference` |
| 监督 | 纯物理约束 | 数据 + 物理混合 |

这两个示例配合起来，基本就能覆盖 PhysicsNeMo 里最常见的 physics-informed 训练范式。

---

## 22. 你迁移到轨道车辆问题时可直接借鉴的部分

如果你要把这套写法迁移到轨道车辆动力学：

1. 用你自己的几何/采样策略替换 `Rectangle + GeometryDatapipe`
2. 用你自己的动力学残差替换 `NavierStokes`
3. 保留“边界/约束损失 + PDE/平衡残差损失”总框架

抽象公式保持不变：

$$
\mathcal{L}_{total}=\mathcal{L}_{equation}+\mathcal{L}_{boundary}+\mathcal{L}_{constraint}
$$

---

## 23. 一句话结论

`ldc_pinns/train.py` 是 PhysicsNeMo 中非常标准、非常“教科书化”的 PINN 训练模板：

> 先定义几何和 PDE，再采样点，再由网络输出构造边界与方程残差，最后统一反向传播。

如果你把这个模板吃透，再结合 `darcy_physics_informed_fno.py`，你就能同时掌握 PINN 与 PINO 两条主线。