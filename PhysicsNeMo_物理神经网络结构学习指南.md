# PhysicsNeMo 物理神经网络结构学习指南

## 1. 这份文档是做什么的

这份文档面向刚开始阅读 `NVIDIA/physicsnemo` 仓库的学习者，目标不是覆盖仓库的全部内容，而是帮助你抓住这套框架里最核心的几个问题：

1. **PhysicsNeMo 到底是什么？**
2. **它的“物理神经网络”是怎么组织的？**
3. **PINN、PINO、Neural Operator、GNN 在这个仓库里分别放在哪里？**
4. **如果我想从源码理解它，应该按什么顺序读？**
5. **如果我想把它和自己当前的轨道/车辆动力学问题联系起来，应当重点看什么？**

这份文档基于当前工作区中的 `physicsnemo` 仓库源码整理，重点参考了：

- `README.md`
- `FAQ.md`
- `physicsnemo/core/module.py`
- `physicsnemo/models/*`
- `docs/examples_introductory.rst`
- `examples/cfd/darcy_fno/*`
- `examples/cfd/darcy_physics_informed/*`
- `examples/cfd/ldc_pinns/*`

---

## 2. 一句话理解 PhysicsNeMo

**PhysicsNeMo 是一个面向科学计算与工程问题的深度学习框架。**

它既不是单纯的神经网络模型库，也不是单纯的 PDE 求解器，而是一个把下面几类能力组合到一起的系统：

- **高性能深度学习训练框架**（基于 PyTorch）
- **面向物理场问题的模型库**（FNO、AFNO、MeshGraphNet、Transolver、Diffusion 等）
- **面向科学数据的数据管道**（网格、网格图、时空场、点云等）
- **物理约束注入机制**（通过 `physicsnemo.sym` 把 PDE、几何、边界条件、残差损失接入训练）

所以，PhysicsNeMo 的核心思想不是“只提供某一种 PINN 网络”，而是：

> **把“神经网络结构”与“物理约束方式”拆开，让用户可以自由组合。**

这也是为什么你在这个仓库里会同时看到：

- 纯数据驱动模型
- 数据 + 物理约束的混合模型
- 纯物理驱动的 PINN
- 神经算子（Neural Operator）
- 图神经网络（GNN）
- 扩散模型（Diffusion）

---

## 3. 理解仓库结构：先把“大楼结构”看清楚

从学习角度，`physicsnemo` 可以拆成 4 层：

### 第 1 层：基础框架层

位于：

- `physicsnemo/core/`
- `physicsnemo/utils/`
- `physicsnemo/distributed/`

这一层负责：

- 模型基类
- checkpoint 保存与加载
- 模型注册机制
- 分布式训练
- logging
- 训练包装器

这一层本身**不等于物理神经网络**，但它是整个系统运行的地基。

### 第 2 层：模型结构层

位于：

- `physicsnemo/models/`

这里存放各种神经网络结构，包括：

- `fno/`：Fourier Neural Operator
- `afno/`：Adaptive Fourier Neural Operator
- `meshgraphnet/`：基于图的物理场建模
- `transolver/`：面向 PDE 的 transformer 风格模型
- `diffusion_unets/`、`dit/`：扩散类模型
- `mlp/`：多层感知机

这一层回答的是：

> **“用什么网络来表示输入到输出的映射？”**

### 第 3 层：数据表达层

位于：

- `physicsnemo/datapipes/`

这一层负责：

- 场数据读取
- benchmark 数据生成
- 网格 / 网格图 / climate 数据流
- 训练时 batch 的组织

这一层回答的是：

> **“物理问题的数据长什么样，怎么喂给网络？”**

### 第 4 层：物理约束层（关键）

这部分主要来自：

- `physicsnemo.sym`

注意：`physicsnemo.sym` **并不完全包含在这个仓库本体中**，而是作为 PhysicsNeMo Symbolic 的一部分被引用和依赖。

它负责：

- 符号化 PDE 定义
- 几何体定义
- 边界条件 / 初始条件定义
- 物理残差损失计算
- 基于自动微分或数值微分计算导数

这一层回答的是：

> **“训练时如何把物理规律加入 loss？”**

---

## 4. 你应该建立的核心认知：网络结构和物理约束是解耦的

初学者最容易误解的一点是：

> “物理神经网络”是不是指某一种特定网络结构？

在 PhysicsNeMo 里，答案通常是否定的。

更准确地说，物理神经网络由两部分组成：

$$
\text{Physics-informed model} = \text{Neural network architecture} + \text{Physics constraints in loss/training}
$$

也就是说：

- **网络结构**：决定你如何表示函数或算子
- **物理约束**：决定你如何让模型满足 PDE、边界条件、守恒关系等

举几个典型组合：

1. **MLP + PDE 残差 + 边界条件损失**  
   这是最经典的 PINN。

2. **FNO + 数据损失**  
   这是纯数据驱动 Neural Operator。

3. **FNO + 数据损失 + PDE 残差损失**  
   这是 PINO（Physics-Informed Neural Operator）。

4. **MeshGraphNet + 网格物理残差**  
   这是图结构上的物理约束学习。

因此，PhysicsNeMo 最重要的设计思想之一是：

> **同一个模型结构可以从“数据驱动”切换到“物理约束驱动”，关键在训练目标而不只在网络本体。**

---

## 5. 核心基类：`physicsnemo.core.module.Module`

源码位置：

- `physicsnemo/core/module.py`

这是整个模型系统的基础类，几乎所有 PhysicsNeMo 模型都会从这里派生。

它本质上是对 `torch.nn.Module` 的增强，主要增加了几个能力：

### 5.1 模型元数据管理

通过 `meta` 记录模型的特性，例如：

- 是否支持 JIT
- 是否支持 CUDA graphs
- 是否支持 ONNX
- 是否支持 auto-grad / physics-informed 特性

### 5.2 模型可注册

可以通过模型注册表按名称检索模型类，这使得：

- checkpoint 更可移植
- 配置驱动实例化更方便
- 大规模实验管理更规范

### 5.3 可序列化 checkpoint

PhysicsNeMo 使用 `.mdlus` 的模型存储机制，这比简单保存 `state_dict` 更进一步，因为它还保存：

- 类名
- 构造参数
- 嵌套子模块信息

这对于科研代码很重要，因为模型定义常常迭代变动，而 PhysicsNeMo 明确在做“可追溯的模型封装”。

### 5.4 支持将 PyTorch 模型转成 PhysicsNeMo 模型

也就是说，如果你已有一个标准 PyTorch 模型，也可以接入 PhysicsNeMo 的基础设施。

---

## 6. 模型层：`physicsnemo.models` 到底有哪些重点

如果你从“物理神经网络”角度去学，最值得先关注的不是所有模型，而是以下几类。

### 6.1 MLP：经典 PINN 的默认基底

目录：

- `physicsnemo/models/mlp/`

用途：

- 输入通常是空间坐标 `x, y, z, t`
- 输出通常是物理场变量 `u, v, p, T ...`

这是最经典 PINN 的网络形式：

$$
(x,y,t) \rightarrow (u,v,p)
$$

优点：

- 简单直接
- 易于自动微分
- 容易写 PDE 残差

缺点：

- 面对高维场问题时效率较低
- 表达复杂时空映射能力有限

### 6.2 FNO：最值得优先掌握的神经算子

目录：

- `physicsnemo/models/fno/`

FNO 的核心思想不是学一个普通函数，而是学习一个**算子映射**：

$$
\mathcal{G}: a(x) \mapsto u(x)
$$

例如：

- 输入：渗透率场
- 输出：压力场

也就是说，它适合学：

- “一个场 -> 另一个场”
- “一个系数函数 -> PDE 解”

这类问题在科学计算中极其常见，所以 FNO 是 PhysicsNeMo 里最关键的一类模型。

### 6.3 MeshGraphNet：网格/图上的物理建模

目录：

- `physicsnemo/models/meshgraphnet/`

适合：

- 非规则网格
- 有拓扑结构的模拟域
- 基于节点与边传播的物理信息建模

对有限元网格、非结构网格问题尤其重要。

### 6.4 Transolver / Transformer 类 PDE 模型

目录：

- `physicsnemo/models/transolver/`

这类模型尝试用注意力机制处理 PDE 问题，适合你进一步扩展视野时阅读。

### 6.5 Diffusion / Generative 类物理模型

目录：

- `physicsnemo/diffusion/`
- `physicsnemo/models/diffusion_unets/`

它们不是经典 PINN，但在物理场生成、数据补全、逆问题、约束采样方面很重要。

---

## 7. FNO 的结构：为什么它对 PDE 问题这么重要

重点源码：

- `physicsnemo/models/fno/fno.py`

这个文件里定义的 `FNO` 类可以看成三段式结构：

### 7.1 输入端

输入是规则网格上的场，例如 2D：

$$
x \in \mathbb{R}^{B \times C_{in} \times H \times W}
$$

这里：

- `B`：batch size
- `C_in`：输入通道数
- `H, W`：空间网格尺寸

### 7.2 频域编码器 `spec_encoder`

FNO 的关键在这部分：

- 将输入变到 Fourier 空间
- 只保留有限个低频 / 重要模态
- 在频域中做卷积/映射
- 再回到空间域

这一步对应了 FNO 论文中的核心思想：

> 使用谱域表示来学习全局耦合关系。

相比局部卷积，FNO 更适合处理 PDE 解中的**长程依赖**和**全局结构**。

### 7.3 点级解码器 `decoder_net`

在频域特征提取后，模型再通过小型全连接网络，把 latent feature 映射到目标物理量通道。

这意味着 FNO 可以理解为：

$$
\text{Input field} \rightarrow \text{Spectral operator layers} \rightarrow \text{Output field}
$$

### 7.4 FNO 在 PhysicsNeMo 里的意义

在 PhysicsNeMo 中，FNO 并不仅仅是一个模型文件，它实际上代表了一种解决范式：

- 用数据学习 PDE 解算子
- 用物理残差去正则化算子学习
- 在低样本条件下通过 PDE loss 提升泛化能力

这就是后面 PINO 的基础。

---

## 8. 三条最关键的学习示例线路

官方文档里给出的入门顺序非常合理，可以概括为三步。

### 8.1 第一步：纯数据驱动 FNO

示例目录：

- `examples/cfd/darcy_fno/`

关键脚本：

- `train_fno_darcy.py`

这个示例的作用是让你先理解：

- FNO 的输入输出长什么样
- datapipe 怎么组织数据
- 一个 PhysicsNeMo 模型如何正常训练

这一步中，训练目标非常简单：

$$
\mathcal{L}_{data} = \mathrm{MSE}(u_{pred}, u_{true})
$$

也就是说，它还没有显式引入 PDE 物理项。

**你应该重点看：**

1. 模型初始化
2. dataloader 的生成方式
3. `forward_train` 如何封装
4. 训练循环和 logger 怎么写

### 8.2 第二步：Darcy PINO / Physics-informed FNO

示例目录：

- `examples/cfd/darcy_physics_informed/`

关键脚本：

- `darcy_physics_informed_fno.py`

这个示例是学习 PhysicsNeMo 物理神经网络结构的**最佳入口之一**。

原因是它清楚展示了以下结构：

1. **网络本体仍然是 FNO**
2. **额外定义 Darcy/Diffusion PDE**
3. **通过 `PhysicsInformer` 计算 PDE residual**
4. **把数据损失和物理损失相加**

总损失可以写成：

$$
\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}
$$

其中：

- `L_data`：预测场和真实场之间的误差
- `L_physics`：PDE 残差
- `\lambda`：物理损失权重

**这是 PhysicsNeMo 里“物理信息如何进入网络训练”的最典型范式。**

### 8.3 第三步：纯 PINN

示例目录：

- `examples/cfd/ldc_pinns/`

关键脚本：

- `train.py`

这是经典 PINN 风格：

- 输入：几何域中的点坐标 `(x, y)`
- 输出：流场变量 `(u, v, p)`
- 训练不依赖真实标签数据
- 完全依赖 PDE 残差 + 边界条件损失

这一类模型通常写作：

$$
(x, y) \rightarrow (u(x,y), v(x,y), p(x,y))
$$

其损失通常是：

$$
\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{BC} + \mathcal{L}_{IC}
$$

在 `ldc_pinns` 里，你会看到：

- 几何采样
- 边界点筛选
- 自动微分求导
- Navier-Stokes 残差构造
- 边界条件项与残差项一起反向传播

如果你能读懂这个例子，就说明你已经真正理解了 PhysicsNeMo 里经典 PINN 的组织方式。

---

## 9. 详细拆解：`darcy_physics_informed_fno.py` 到底做了什么

这是当前最值得细读的脚本之一。

### 9.1 先定义 PDE

脚本里使用了：

- `Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)`

这个定义表达的是一个 2D 稳态扩散 / Darcy 风格方程。

你可以把它理解成：

- `u`：待预测物理量（例如压力）
- `k`：介质参数（例如渗透率）
- `Q`：源项

也就是说，这个 PDE 规定了预测场与输入系数场之间必须满足某种微分关系。

### 9.2 再定义 FNO 模型

模型仍然是：

- `FNO(...)`

这里的关键理解是：

> **PINO 不是“换了个网络”，而是“给已有网络加了物理损失”。**

### 9.3 再定义 `PhysicsInformer`

这是最关键的桥梁层。

`PhysicsInformer` 做的事情可以概括为：

1. 接收 PDE 定义
2. 接收模型输出和需要的物理输入
3. 根据指定导数计算方式生成 PDE 残差
4. 把残差返回给训练脚本

在这个示例里，导数方法是：

- `grad_method="finite_difference"`

这说明它走的是 **PINO 路线**：

- 网络在规则网格上预测整张场
- 对预测场做数值微分
- 得到 PDE 残差

### 9.4 训练循环中的核心流程

每个 batch 基本做了 4 件事：

1. `out = model(invar[:, 0].unsqueeze(dim=1))`  
   用 FNO 预测输出场。

2. `residuals = phy_informer.forward({...})`  
   利用预测场和物理参数场，计算 PDE 残差。

3. `loss_data = mse(outvar, out)`  
   计算数据误差。

4. `loss = loss_data + weight * loss_pde`  
   合成总损失并反向传播。

所以这个脚本体现了物理神经网络训练的经典公式：

$$
\text{Prediction} \rightarrow \text{Residual} \rightarrow \text{Physics loss} \rightarrow \text{Total loss}
$$

### 9.5 这个脚本最值得你学什么

不是“怎么调用一个现成类”，而是以下几个思想：

- **网络结构和 PDE 是分开的**
- **PDE 是通过额外 loss 注入的**
- **数值导数与自动微分是可切换的**
- **PhysicsInformer 是连接网络输出与 PDE 的中间层**

---

## 10. 详细拆解：`ldc_pinns/train.py` 为什么是标准 PINN

这个脚本非常适合拿来理解“纯物理驱动训练”的结构。

### 10.1 网络输入输出

模型使用：

- `FullyConnected(in_features=2, out_features=3, ...)`

表示：

$$
(x,y) \rightarrow (u,v,p)
$$

这是 PINN 的最典型写法。

### 10.2 数据不是标签样本，而是几何域采样点

脚本中使用：

- `Rectangle(...)` 定义几何域
- `GeometryDatapipe(...)` 采样边界点和内部点

也就是说，训练数据不再是 `(input, label)` 形式，而是：

- 边界点
- 内部点
- 几何信息（如 `sdf`）

### 10.3 物理损失构成

训练过程中包含两大类损失：

#### 边界条件损失

例如：

- no-slip 边界速度为 0
- top wall 有指定速度

这会变成 MSE 形式的约束项。

#### PDE 残差损失

通过 `NavierStokes(...)` 定义 PDE，再由 `PhysicsInformer` 计算：

- continuity
- momentum_x
- momentum_y

这些残差被平方求平均后加入总 loss。

### 10.4 为什么这是“纯 PINN”

因为它没有使用真实 CFD 标签场作为监督信号，而是完全依赖：

- 物理方程
- 边界条件
- 几何域采样

这正是经典 PINN 的核心定义。

---

## 11. PINN 与 PINO 在 PhysicsNeMo 里的差异

这部分非常关键，建议你反复记忆。

| 维度 | PINN | PINO |
|---|---|---|
| 常见网络 | MLP | FNO / Neural Operator |
| 输入 | 坐标点 | 输入场 / 系数场 |
| 输出 | 单点物理量 | 整张场 |
| 导数获取 | 自动微分较常见 | 数值微分 / 频域微分较常见 |
| 数据依赖 | 可无监督 | 通常有数据监督 + 物理正则 |
| 典型 loss | PDE + BC + IC | data loss + PDE residual |
| 适合问题 | 小域、解析式友好问题 | 大规模场映射、算子学习 |

从源码学习角度：

- 想学经典物理约束写法，先看 `ldc_pinns`
- 想学 PhysicsNeMo 里更现代、更实用的物理学习方式，优先看 `darcy_physics_informed_fno`

---

## 12. `PhysicsInformer`：最关键的“桥梁组件”

不管是 PINN 还是 PINO，真正把神经网络和 PDE 联系起来的核心角色，都是：

- `PhysicsInformer`

你可以把它理解为一个“物理残差生成器”。

它的输入一般包括：

- 模型预测量，例如 `u, v, p`
- 坐标或网格信息
- PDE 所需的额外物理参数，例如 `k`

它的输出一般包括：

- continuity residual
- momentum residual
- diffusion residual
- 其他 PDE 项

它的本质作用是：

$$
\text{Neural prediction} \rightarrow \text{Differentiation} \rightarrow \text{PDE residual tensor}
$$

它使得训练脚本不需要自己手写大量繁琐导数逻辑。

---

## 13. `physicsnemo.sym` 的定位：为什么它几乎等于“PhysicsNeMo 的物理大脑”

如果只看 `physicsnemo` 主仓库，你会发现很多地方直接导入：

- `physicsnemo.sym.eq.pdes.*`
- `physicsnemo.sym.geometry.*`
- `physicsnemo.sym.eq.phy_informer.*`

这说明：

- `physicsnemo` 主仓库主要提供模型、训练工具、datapipe、基础设施
- `physicsnemo.sym` 负责把“符号化物理知识”引入训练过程

因此从学习角度要记住：

> **PhysicsNeMo Core 负责“怎么训练网络”，PhysicsNeMo Sym 负责“怎么把物理写进训练”。**

如果你未来要写自定义 PDE、自定义边界条件、自定义几何域，最终一定会深入到 `physicsnemo.sym`。

---

## 14. 推荐阅读顺序（非常重要）

下面给出一条适合工程背景学习者的阅读顺序。

### 第一阶段：先建立整体框架感

按这个顺序读：

1. `physicsnemo/README.md`
2. `physicsnemo/FAQ.md`
3. `physicsnemo/docs/api/models/modules.rst`

目标：

- 知道 PhysicsNeMo 是做什么的
- 知道 Core 和 Sym 的区别
- 知道模型 zoo 的范围

### 第二阶段：理解模型底座

按这个顺序读：

1. `physicsnemo/core/module.py`
2. `physicsnemo/models/__init__.py`
3. `physicsnemo/models/fno/fno.py`

目标：

- 理解模型的统一基类
- 理解 FNO 的输入输出结构

### 第三阶段：从纯数据到物理约束

按这个顺序读：

1. `examples/cfd/darcy_fno/train_fno_darcy.py`
2. `examples/cfd/darcy_physics_informed/darcy_physics_informed_fno.py`
3. `examples/cfd/ldc_pinns/train.py`

目标：

- 先看纯数据驱动训练
- 再看数据 + 物理混合训练
- 最后看纯物理驱动训练

### 第四阶段：扩展到更复杂的 PINO

推荐继续看：

- `examples/cfd/swe_nonlinear_pino/`
- `examples/cfd/mhd_pino/`

目标：

- 理解多方程耦合 residual
- 理解时空 PDE 的 physics-informed 训练

---

## 15. 如果把它映射到你当前的轨道车辆系统问题，应怎么理解

你当前工作区是轨道/车辆动力学仿真方向，这和 PhysicsNeMo 的常见 CFD 示例不同，但底层思想是可以迁移的。

### 15.1 如果你想做“响应预测器”

例如：

- 输入：轨道不平顺、缺陷参数、车辆参数、速度
- 输出：轮轨力、加速度、位移、脱轨系数等

那么你可以把问题理解为：

$$
\text{system condition field/parameter} \rightarrow \text{response field/time series}
$$

这类问题很适合借鉴：

- FNO
- Transolver
- 时空 operator 网络

### 15.2 如果你想引入“动力学方程约束”

例如：

- 质量-阻尼-刚度系统平衡
- 轨道梁/车辆耦合运动方程
- 能量守恒 / 动量平衡

那你要做的事情就和 PhysicsNeMo 里的 PINN/PINO 很像：

- 用神经网络预测状态变量
- 用动力学方程残差构造 `physics loss`

### 15.3 如果你的数据是规则时空网格

例如：

- 位置 × 时间 的位移/加速度场

可以优先考虑：

- FNO / 时空算子网络

### 15.4 如果你的数据是结构图或离散自由度网络

例如：

- 车辆系统多刚体节点
- 轨道离散单元之间的连接关系

可以考虑：

- GNN / MeshGraphNet 风格模型

换句话说，你不需要生搬硬套 CFD 示例里的 PDE 形式，但它们提供了非常重要的模板：

> **如何把“已有的仿真结构知识”写成损失或约束，再和神经网络结合。**

---

## 16. 学习 PhysicsNeMo 时最容易混淆的几个点

### 16.1 “PhysicsNeMo 是否等于 PINN 框架？”

不是。

它包含 PINN，但远不止 PINN。

### 16.2 “FNO 本身是不是 physics-informed？”

不一定。

FNO 本身只是神经算子结构。只有当你在 loss 中加入 PDE / 守恒 / 边界条件时，它才成为 physics-informed 模型。

### 16.3 “为什么主仓库里很多 PINN 能力来自 `physicsnemo.sym`？”

因为 PhysicsNeMo 的架构就是把：

- 高性能深度学习核心
- 符号物理约束能力

拆开设计的。

### 16.4 “是不是必须有真实数据才能训练？”

不一定。

- 纯 PINN：可以只靠 PDE + BC + IC
- PINO：常常是数据监督 + 物理损失混合

---

## 17. 建议你接下来怎么学

如果你的目标是“真正看懂 PhysicsNeMo 的物理神经网络结构”，建议按下面方式推进。

### 路线 A：先懂结构

适合第一次阅读：

1. 看 `README` 和 `FAQ`
2. 看 `fno.py`
3. 看 `darcy_fno`
4. 看 `darcy_physics_informed_fno.py`
5. 看 `ldc_pinns/train.py`

### 路线 B：先懂物理约束怎么接进去

适合你当前最关心“如何写 physics loss”的需求：

1. 直接读 `darcy_physics_informed_fno.py`
2. 再读 `ldc_pinns/train.py`
3. 再回头读 `fno.py`

### 路线 C：面向你自己的项目迁移

适合未来做轨道车辆方向应用：

1. 先抽象你自己的输入/输出张量结构
2. 判断问题更像 PINN、PINO、还是 operator learning
3. 按 PhysicsNeMo 示例改写成最小实验

---

## 18. 最终总结

如果只用一句话总结 PhysicsNeMo 的物理神经网络结构，可以这样说：

> **PhysicsNeMo 用 `core + models + datapipes` 搭建神经网络训练骨架，再通过 `physicsnemo.sym` 把 PDE、几何和边界条件注入到损失函数中，从而实现 PINN、PINO 和更广义的 Physics AI。**

从入门顺序上，最推荐你掌握的三个示例是：

1. `examples/cfd/darcy_fno/`：理解纯数据驱动神经算子
2. `examples/cfd/darcy_physics_informed/`：理解 PINO
3. `examples/cfd/ldc_pinns/`：理解经典 PINN

如果你把这三个例子真正读透，那么你对 PhysicsNeMo 的理解就不再停留在“会运行示例”，而是进入了“知道它如何组织物理神经网络”的层面。

---

## 19. 后续建议（可作为下一步工作）

你接下来可以继续做下面三件事中的任意一个：

### 选项 1：逐行精读 `darcy_physics_informed_fno.py`

目标：完全理解 PINO 训练循环。

### 选项 2：梳理 `physicsnemo.sym` 中 PDE 与 `PhysicsInformer` 的调用链

目标：弄清楚 Physics loss 是如何从符号方程变成张量残差的。

### 选项 3：把你的轨道车辆问题映射成 PhysicsNeMo 风格任务

目标：设计一个你自己的最小 physics-informed 学习原型。

如果你愿意，下一步我可以继续帮你写第二份文档：

- **《darcy_physics_informed_fno.py 逐行源码讲解》**

或者直接帮你画一份：

- **《PhysicsNeMo 物理神经网络结构图（Core / Models / Sym / Loss）》**