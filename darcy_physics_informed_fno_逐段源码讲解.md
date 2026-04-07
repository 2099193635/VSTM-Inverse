# `darcy_physics_informed_fno.py` 逐段源码讲解

## 1. 这份文档的目标

这份文档专门讲解以下脚本：

- `physicsnemo/examples/cfd/darcy_physics_informed/darcy_physics_informed_fno.py`

它是一个非常典型的 **PINO（Physics-Informed Neural Operator）** 训练脚本，展示了如何把：

- `FNO` 这种神经算子模型
- 数据监督损失
- PDE 残差损失

组合成一个完整的 physics-informed 训练流程。

如果你已经读过根目录下的 [PhysicsNeMo_物理神经网络结构学习指南.md](PhysicsNeMo_物理神经网络结构学习指南.md)，那么这份文档就是它的“源码落地版”。

---

## 2. 先看这个脚本在做什么

用一句话概括，这个脚本做的是：

> 用 `FNO` 预测 Darcy 流问题中的压力场，同时通过 `PhysicsInformer` 计算 PDE 残差，把数据损失和物理损失一起训练。

更具体一点：

1. 从 HDF5 数据集读入 Darcy 问题样本
2. 用 `FNO` 从输入系数场预测输出解场
3. 用 `Diffusion` PDE + `PhysicsInformer` 计算残差
4. 构造总损失
5. 反向传播并保存 checkpoint

这个脚本不是纯 PINN，而是：

$$
\text{PINO} = \text{Operator network} + \text{data loss} + \text{physics residual loss}
$$

---

## 3. 先建立整体执行流程图

你可以先把这个脚本理解成下面这条流水线：

$$
\text{HDF5 input} \rightarrow \text{FNO} \rightarrow \text{predicted field} \rightarrow \text{PhysicsInformer} \rightarrow \text{PDE residual}
$$

然后总损失为：

$$
\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}
$$

其中：

- $\mathcal{L}_{data}$：预测场与真实场的误差
- $\mathcal{L}_{physics}$：Darcy / Diffusion PDE 的残差
- $\lambda$：物理项权重

---

## 4. 第一部分：导入模块

脚本开头导入了这些库：

- `hydra`
- `matplotlib`
- `numpy`
- `torch`
- `torch.nn.functional as F`
- `LaunchLogger`
- `save_checkpoint`
- `FNO`
- `Diffusion`
- `PhysicsInformer`
- `DataLoader`
- `HDF5MapStyleDataset`

这一部分可以按“职责”来理解。

### 4.1 训练与数值计算相关

- `torch`
- `torch.nn.functional as F`
- `numpy`

负责前向、损失、反向传播和一些数组处理。

### 4.2 配置与路径管理

- `hydra`
- `to_absolute_path`
- `DictConfig`

负责读取配置文件、管理运行目录、把相对路径转成绝对路径。

### 4.3 PhysicsNeMo 模型与物理接口

- `FNO`
- `Diffusion`
- `PhysicsInformer`

这是这个脚本最关键的三件套：

- `FNO`：网络结构
- `Diffusion`：PDE 定义
- `PhysicsInformer`：把预测结果变成 PDE 残差

### 4.4 日志与存储

- `LaunchLogger`
- `save_checkpoint`

分别负责：

- 训练过程日志
- 每个 epoch 的模型保存

### 4.5 自定义数据集类

- `from utils import HDF5MapStyleDataset`

这个类在同目录的 `utils.py` 中定义，负责把 Darcy 的 HDF5 数据读成训练样本。

---

## 5. 第二部分：`validation_step` 做了什么

脚本首先定义了一个函数：

- `validation_step(model, dataloader, epoch)`

它不是训练主循环的一部分，而是每个 epoch 结束后做验证和可视化。

### 5.1 `model.eval()`

```python
model.eval()
```

这表示模型切到评估模式。

对于某些含有：

- dropout
- batch normalization

的网络，这一步会改变行为。虽然 FNO 这里不一定强依赖这一步，但保持评估模式是标准写法。

### 5.2 `torch.no_grad()`

```python
with torch.no_grad():
```

表示验证阶段不记录梯度，从而：

- 节省显存
- 加快推理
- 避免无意义的反向图构建

### 5.3 取数据并前向预测

```python
for data in dataloader:
    invar, outvar, _, _ = data
    out = model(invar[:, 0].unsqueeze(dim=1))
```

这里有几个关键点。

#### 关键点 1：`data` 返回 4 项

来自 `HDF5MapStyleDataset.__getitem__`，每个样本返回：

1. `invar`
2. `outvar`
3. `x_invar`
4. `y_invar`

但这个脚本里实际只使用了前两项。

#### 关键点 2：为什么只取 `invar[:, 0]`

在 `utils.py` 中，`invar` 由三部分拼成：

- `Kcoeff`
- `Kcoeff_x`
- `Kcoeff_y`

即一共 3 个通道。

但这里实际送进 FNO 的是：

```python
invar[:, 0].unsqueeze(dim=1)
```

也就是只取第 1 个通道 `Kcoeff`，重新补回 channel 维度后，变成：

$$
[B,1,H,W]
$$

这说明：

- 模型输入是渗透率场 `k`
- `Kcoeff_x` 和 `Kcoeff_y` 并没有直接作为网络输入进入 FNO

### 5.4 计算验证误差

```python
loss_epoch += F.mse_loss(outvar, out)
```

这里验证误差只看数据项：

$$
\mathcal{L}_{val} = \mathrm{MSE}(u_{pred}, u_{true})
$$

也就是说，验证阶段并没有再算 PDE 残差。

这很常见，因为：

- 验证主要关心预测质量
- PDE 残差更多用于训练正则化

### 5.5 画图保存

函数后半段把最后一个 batch 的结果画出来：

- 真值图 `True`
- 预测图 `Pred`
- 误差图 `Difference`

保存为：

- `results_{epoch}.png`

这对观察训练过程是否收敛很有帮助。

### 5.6 `validation_step` 的本质作用

这个函数一共做了两件事：

1. 统计验证集上的 MSE
2. 画一张本 epoch 的预测可视化图

---

## 6. 第三部分：Hydra 主函数入口

主函数入口是：

```python
@hydra.main(version_base="1.3", config_path="conf", config_name="config_pino.yaml")
def main(cfg: DictConfig):
```

这里说明这个脚本不是“硬编码参数”，而是通过 Hydra 读取配置文件。

对应配置文件在：

- `physicsnemo/examples/cfd/darcy_physics_informed/conf/config_pino.yaml`

这个配置文件里最关键的参数是：

- 学习率 `start_lr`
- 学习率衰减系数 `gamma`
- epoch 数 `max_epochs`
- 物理损失权重 `physics_weight`
- FNO 的结构超参数

### 6.1 为什么要用 Hydra

Hydra 的好处是：

- 训练超参数统一管理
- 输出目录自动组织
- 便于命令行覆盖配置
- 多实验比较更方便

此外，这个配置里还设置了：

```yaml
hydra:
  job:
    chdir: True
  run:
    dir: ./outputs_pino
```

含义是：

- 运行时工作目录切到 `outputs_pino`
- 输出文件会集中放在这个目录下

---

## 7. 第四部分：设备选择

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

这部分很直接：

- 有 GPU 就用 GPU
- 否则用 CPU

这个 `device` 后面会同时传给：

- 数据集
- 模型
- `PhysicsInformer`

这样可以保持张量和模型在同一个设备上。

---

## 8. 第五部分：初始化日志系统

```python
LaunchLogger.initialize()
```

这一步是 PhysicsNeMo 的日志系统初始化。

后面训练和验证都用：

- `with LaunchLogger(...) as log:`

来做日志记录。

这样做的好处是：

- 控制台输出更规整
- 支持 mini-batch 和 epoch 级别记录
- 方便接入实验管理工具

---

## 9. 第六部分：定义 Darcy PDE

这里是整个脚本最核心的物理定义之一：

```python
forcing_fn = 1.0 * 4.49996e00 * 3.88433e-03
darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)
```

### 9.1 这里为什么用 `Diffusion`

Darcy 流问题在很多形式下可以写成扩散型方程，因此示例这里直接复用了 `Diffusion` PDE 类。

你可以把它理解成：

$$
\nabla \cdot (k \nabla u) = Q
$$

其中：

- `u`：待预测的场
- `k`：介质系数场
- `Q`：源项

### 9.2 参数含义

- `T="u"`：表示主变量名是 `u`
- `time=False`：说明是稳态问题，不含时间维度
- `dim=2`：二维空间问题
- `D="k"`：扩散系数名是 `k`
- `Q=forcing_fn`：源项常数

### 9.3 为什么要先定义 PDE 再训练

因为后续 `PhysicsInformer` 需要知道：

- 方程的结构是什么
- 要对哪些变量求导
- 最终要输出什么残差名称

也就是说，`Diffusion(...)` 是“物理规则的声明”，而不是训练本身。

---

## 10. 第七部分：构造数据集

脚本中：

```python
dataset = HDF5MapStyleDataset(
    to_absolute_path("./datasets/Darcy_241/train.hdf5"), device=device
)
validation_dataset = HDF5MapStyleDataset(
    to_absolute_path("./datasets/Darcy_241/validation.hdf5"), device=device
)
```

### 10.1 `to_absolute_path`

因为 Hydra 会切换运行目录，所以相对路径可能失效。

`to_absolute_path` 的作用就是把相对路径转换成稳定的绝对路径。

### 10.2 `HDF5MapStyleDataset` 的输出格式

这个类在 `utils.py` 里定义，单个样本返回：

```python
return invar, outvar, x_invar, y_invar
```

其中：

#### `invar`

拼接自：

- `Kcoeff`
- `Kcoeff_x`
- `Kcoeff_y`

所以通道数为 3。

#### `outvar`

对应：

- `sol`

即真实解场。

#### `x_invar`, `y_invar`

是规则网格上的坐标张量。

### 10.3 这个数据集有一个非常值得注意的点

在 `utils.py` 中，数据被做了缩放：

- `Kcoeff / 4.49996e00`
- `sol / 3.88433e-03`

因此，脚本中的 `forcing_fn = 1.0 * 4.49996e00 * 3.88433e-03` 实际上是在补偿这种数据缩放，使 PDE 仍然和归一化后的量保持一致。

这是这个示例非常重要但很容易忽略的细节。

---

## 11. 第八部分：DataLoader

```python
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
```

这里设置得很简单：

- 训练集 batch size = 1
- 验证集 batch size = 1

### 11.1 为什么可能用 batch size = 1

这种 PDE 场数据每个样本本身就很大，比如：

- 通道 × 240 × 240

同时还要算 PDE 残差，显存消耗并不小，所以示例采取较保守的 batch size。

### 11.2 `shuffle` 的区别

- 训练集：`shuffle=True`
- 验证集：`shuffle=False`

符合标准训练习惯。

---

## 12. 第九部分：定义 FNO 模型

```python
model = FNO(
    in_channels=cfg.model.fno.in_channels,
    out_channels=cfg.model.fno.out_channels,
    decoder_layers=cfg.model.fno.decoder_layers,
    decoder_layer_size=cfg.model.fno.decoder_layer_size,
    dimension=cfg.model.fno.dimension,
    latent_channels=cfg.model.fno.latent_channels,
    num_fno_layers=cfg.model.fno.num_fno_layers,
    num_fno_modes=cfg.model.fno.num_fno_modes,
    padding=cfg.model.fno.padding,
).to(device)
```

对应配置文件中的参数是：

- `in_channels: 1`
- `out_channels: 1`
- `dimension: 2`
- `latent_channels: 32`
- `num_fno_layers: 4`
- `num_fno_modes: 12`
- `padding: 9`

### 12.1 这意味着什么

模型接收：

$$
[B,1,H,W]
$$

输出：

$$
[B,1,H,W]
$$

也就是说，FNO 在这里学习的是：

$$
k(x,y) \mapsto u(x,y)
$$

这正是 Darcy 型问题里最典型的“系数场到解场”的映射。

### 12.2 为什么这很适合 FNO

因为 FNO 本来就是为学习 PDE 解算子而设计的。

相比传统 PINN 的逐点拟合，FNO 更适合：

- 规则网格上的整场预测
- 大规模场映射
- 低频主导的 PDE 解结构

---

## 13. 第十部分：定义 `PhysicsInformer`

```python
phy_informer = PhysicsInformer(
    required_outputs=["diffusion_u"],
    equations=darcy,
    grad_method="finite_difference",
    device=device,
    fd_dx=1 / 240,
)
```

这部分是 physics-informed 训练的核心桥梁。

### 13.1 `required_outputs=["diffusion_u"]`

说明我们需要的 PDE 输出项名称叫：

- `diffusion_u`

也就是扩散方程对变量 `u` 的残差。

### 13.2 `equations=darcy`

说明残差的计算规则来自前面定义的那个 `Diffusion` 方程。

### 13.3 `grad_method="finite_difference"`

这是最关键的地方之一。

它表示导数不是通过自动微分来算，而是通过：

- 有限差分

来算。

这也是为什么这个例子属于 **PINO 风格**，而不是经典 MLP-PINN 风格。

#### PINN 常见方式

- 输入是坐标点
- 输出是点值
- 导数常用自动微分

#### PINO 常见方式

- 输入是整张场
- 输出是整张场
- 导数常用有限差分 / Fourier 导数

### 13.4 `fd_dx = 1 / 240`

这表示网格间距，来自单位正方形被离散成 240 个点这一设定。

也就是：

$$
\Delta x = \frac{1}{240}
$$

如果这个值设置错误，PDE 残差的数值尺度就会不对。

---

## 14. 第十一部分：优化器与学习率调度器

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    betas=(0.9, 0.999),
    lr=cfg.start_lr,
    weight_decay=0.0,
)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
```

配置文件对应：

- `start_lr: 0.001`
- `gamma: 0.99948708`

### 14.1 Adam 的作用

这是最常见的深度学习优化器，适合这个示例。

### 14.2 指数衰减学习率

每步更新后学习率按固定比例衰减：

$$
lr_{t+1} = \gamma \cdot lr_t
$$

这样做的目的是：

- 前期较快搜索
- 后期更稳定收敛

值得注意的是，这个脚本把 `scheduler.step()` 放在每个 mini-batch 后，因此衰减发生得比较细粒度。

---

## 15. 第十二部分：主训练循环

训练循环从这里开始：

```python
for epoch in range(cfg.max_epochs):
```

配置里：

- `max_epochs: 50`

每个 epoch 内部又包了一层：

```python
with LaunchLogger("train", ...) as log:
```

说明训练日志是按 epoch 组织的。

---

## 16. 第十三部分：每个 batch 的训练步骤

这一段是整个脚本最关键的逻辑核心。

### 16.1 清空梯度

```python
optimizer.zero_grad()
```

标准 PyTorch 写法，避免梯度累积。

### 16.2 取出输入输出

```python
invar = data[0]
outvar = data[1]
```

只取了数据集返回的前两项：

- 输入场
- 真值输出场

### 16.3 前向预测

```python
out = model(invar[:, 0].unsqueeze(dim=1))
```

这一步做的是：

$$
u_{pred} = \text{FNO}(k)
$$

即从输入系数场 `k` 预测解场 `u`。

### 16.4 计算 PDE 残差

```python
residuals = phy_informer.forward(
    {
        "u": out,
        "k": invar[:, 0:1],
    }
)
pde_out_arr = residuals["diffusion_u"]
```

这里非常关键。

`PhysicsInformer` 接收一个字典，其中：

- `u`：模型预测的输出场
- `k`：输入的物理系数场

然后根据 `Diffusion` PDE 计算出：

- `diffusion_u` 残差

也就是离散网格上每个点的 PDE 不满足程度。

### 16.5 为什么要裁掉边界再 pad 回去

```python
pde_out_arr = F.pad(
    pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
)
```

这是一个很值得注意的数值处理细节。

#### 原因

有限差分在边界附近通常不够稳定，或者 stencil 不完整。

所以这里做了两步：

1. 先取内部区域 `2:-2, 2:-2`
2. 再把外侧 pad 回 0

这样相当于：

- 只信任内部区域的 PDE 残差
- 边界附近不参与物理损失计算

这是物理信息学习里很常见的技巧。

### 16.6 计算 PDE 损失

```python
loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))
```

这里用的是 L1 损失，而不是平方损失。

也就是：

$$
\mathcal{L}_{physics} = \|R(u,k)\|_1
$$

其中：

- $R(u,k)$ 表示 PDE 残差

这意味着训练目标是让残差尽量接近 0。

### 16.7 计算数据损失

```python
loss_data = F.mse_loss(outvar, out)
```

这就是标准监督项：

$$
\mathcal{L}_{data} = \mathrm{MSE}(u_{true}, u_{pred})
$$

### 16.8 组合总损失

```python
loss = loss_data + 1 / 240 * cfg.physics_weight * loss_pde
```

这里总损失是：

$$
\mathcal{L} = \mathcal{L}_{data} + \frac{1}{240} \cdot \lambda \cdot \mathcal{L}_{physics}
$$

其中配置里：

- `physics_weight = 0.1`

所以真正使用的是：

$$
\lambda_{eff} = \frac{0.1}{240}
$$

### 16.9 为什么还要乘 `1/240`

这通常是在做尺度平衡。

因为：

- PDE 残差的量纲和数值尺度
- 数据 MSE 的数值尺度

可能差很多。

通过乘上网格尺度 `1/240`，脚本在尝试把物理损失调到一个更合适的数量级。

### 16.10 反向传播与参数更新

```python
loss.backward()
optimizer.step()
scheduler.step()
```

标准三步：

1. 反向传播
2. 更新参数
3. 更新学习率

### 16.11 日志记录

```python
log.log_minibatch(
    {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach()}
)
```

这里把两个损失分开记录，而不是只记总损失。

这样做特别重要，因为你可以看到训练时：

- 数据误差是否在降
- PDE 残差是否也在降
- 是否存在两者冲突

---

## 17. 第十四部分：epoch 级别日志

```python
log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
```

这一句每个 epoch 结束时记录学习率，方便观察 scheduler 是否按预期工作。

---

## 18. 第十五部分：验证阶段

训练完一个 epoch 后，脚本执行：

```python
with LaunchLogger("valid", epoch=epoch) as log:
    error = validation_step(model, validation_dataloader, epoch)
    log.log_epoch({"Validation error": error})
```

这里逻辑非常清晰：

1. 进入验证日志上下文
2. 调用 `validation_step`
3. 记录验证误差

注意这里的验证误差是数据 MSE，不是 PDE loss。

---

## 19. 第十六部分：保存 checkpoint

```python
save_checkpoint(
    "./checkpoints",
    models=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
)
```

每个 epoch 都会保存：

- 模型参数
- 优化器状态
- scheduler 状态
- 当前 epoch

这样可以：

- 中断后恢复训练
- 保存不同阶段模型
- 后续单独做推理或对比

---

## 20. 第十七部分：脚本的输入输出张量关系

这一节非常适合你在脑中建立“张量流动图”。

### 20.1 数据集返回

单个样本：

- `invar`: `[3, 240, 240]`
- `outvar`: `[1, 240, 240]`
- `x_invar`: `[1, 240, 240]`
- `y_invar`: `[1, 240, 240]`

### 20.2 DataLoader 后

batch size = 1 时：

- `invar`: `[1, 3, 240, 240]`
- `outvar`: `[1, 1, 240, 240]`

### 20.3 送进 FNO 的输入

```python
invar[:, 0].unsqueeze(dim=1)
```

得到：

- `[1, 1, 240, 240]`

### 20.4 FNO 输出

- `out`: `[1, 1, 240, 240]`

### 20.5 PDE 残差

- `pde_out_arr`: `[1, 1, 240, 240]`

### 20.6 损失级别

- `loss_data`: 标量
- `loss_pde`: 标量
- `loss`: 标量

这就是整个训练图的张量主线。

---

## 21. 这个脚本为什么是典型 PINO，而不是 PINN

这是一个非常重要的判断题。

### 21.1 它不是 PINN 的原因

因为它并没有：

- 直接把坐标点 `(x, y)` 喂给 MLP
- 通过自动微分直接对坐标求导
- 只靠 PDE 与边界条件训练

### 21.2 它是 PINO 的原因

因为它做的是：

- 用 FNO 学习场到场映射
- 对输出场用数值微分计算 PDE 残差
- 同时保留数据监督项

这正是 PINO 的典型结构。

换句话说，它更像：

$$
\text{coefficient field} \rightarrow \text{solution field}
$$

而不是：

$$
(x,y) \rightarrow u(x,y)
$$

---

## 22. 这个示例里最重要的 8 个学习点

如果你只抓重点，建议记住下面 8 点。

### 22.1 物理规则先于训练被声明

通过 `Diffusion(...)` 先把 PDE 明确写出来。

### 22.2 模型结构和物理约束是分离的

`FNO` 负责预测，`PhysicsInformer` 负责残差。

### 22.3 物理信息是通过 loss 注入的

不是改网络结构，而是改训练目标。

### 22.4 `PhysicsInformer` 是核心桥梁

没有它，脚本就只是一个普通 FNO 监督学习脚本。

### 22.5 这里用的是有限差分，而不是自动微分

这决定了它是 PINO 风格。

### 22.6 边界残差被裁掉

说明数值微分在边界区域要特别小心。

### 22.7 数据缩放会影响 PDE 系数

`forcing_fn` 的尺度并不是随手写的，它和数据归一化直接相关。

### 22.8 训练时同时监控 `loss_data` 和 `loss_pde`

这有助于判断模型是在“学数据”还是“学物理”。

---

## 23. 如果你想把这个模板迁移到自己的问题，应该怎么改

如果你未来想把这个脚本借鉴到你的轨道车辆动力学任务，可以按下面思路映射。

### 23.1 替换输入输出定义

当前脚本是：

- 输入：渗透率场 `k(x,y)`
- 输出：压力场 `u(x,y)`

你可以改成：

- 输入：轨道不平顺场 / 缺陷参数场 / 工况参数
- 输出：位移场 / 加速度场 / 轮轨力时空分布

### 23.2 替换 PDE / 动力学约束

当前使用的是 `Diffusion`。

你未来可以替换为：

- 结构动力学平衡方程
- 车辆-轨道耦合方程
- 振动传播方程

### 23.3 替换导数方式

如果你的输出是规则时空网格，可以继续用：

- 有限差分
- 频域导数

如果你更偏坐标点式表达，则可能转回：

- 自动微分

### 23.4 保留这个总框架

无论具体问题怎么变，整体结构通常仍可保留：

$$
\text{Network} + \text{Physics residual} + \text{Data loss}
$$

这也是这个示例最值得借鉴的部分。

---

## 24. 你下一步最建议做什么

如果你已经能看懂这份文档，下一步最推荐做的是下面三项之一。

### 选项 1：对照源码逐段阅读

打开：

- `physicsnemo/examples/cfd/darcy_physics_informed/darcy_physics_informed_fno.py`

边看源码边对照这份文档。

### 选项 2：继续读 `utils.py`

重点理解：

- 数据归一化是怎么做的
- 为什么 `forcing_fn` 需要缩放补偿

### 选项 3：再读 `ldc_pinns/train.py`

对比理解：

- PINO：场到场 + 有限差分残差
- PINN：点到值 + 自动微分残差

---

## 25. 最后的总结

`darcy_physics_informed_fno.py` 之所以是 PhysicsNeMo 学习路径里的关键示例，是因为它把下面几件事清楚地结合在一起：

1. `FNO` 负责学习 Darcy 解算子
2. `Diffusion` 负责声明 PDE 结构
3. `PhysicsInformer` 负责把预测场变成 PDE 残差
4. `loss_data + loss_pde` 负责把数据与物理统一到训练目标里

如果你已经理解这四点，那么你实际上已经抓住了 PhysicsNeMo 中一大类物理神经网络的核心设计模式。

---

## 26. 可继续补充的后续文档

如果你愿意，下一步还可以继续补下面两类文档：

1. **`ldc_pinns/train.py` 逐段源码讲解**
2. **PhysicsInformer / Diffusion / FNO 三者调用链分析**

如果你的目标是最终迁移到你自己的轨道车辆系统，我更建议优先做第 2 个，因为那会直接帮助你设计自己的 `physics loss`。