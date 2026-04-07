# 轨道车辆问题 → PhysicsNeMo/PINN/PINO 任务改写模板

## 1. 文档目标

这份文档给你一个可直接执行的迁移模板：

- 把你当前 `VTCM_PYTHON` 项目的仿真问题
- 改写为可用 PhysicsNeMo 训练的任务

重点是“怎么落地”，不是纯概念介绍。

---

## 2. 先给出总框架

对轨道车辆系统，建议先按三种任务形态来拆：

1. **纯数据驱动（Operator Learning）**
2. **数据 + 物理约束（PINO）**
3. **纯物理约束（PINN）**

统一写法：

$$
\mathcal{L}_{total}=\alpha\mathcal{L}_{data}+\beta\mathcal{L}_{physics}+\gamma\mathcal{L}_{constraint}
$$

其中：

- $\mathcal{L}_{data}$：仿真/测量真值误差
- $\mathcal{L}_{physics}$：动力学方程残差
- $\mathcal{L}_{constraint}$：边界、初值、工程约束（例如限值）

---

## 3. 与你当前工程目录的映射建议

你当前项目关键目录可映射如下：

- [configs](configs) → 训练配置、变量定义、损失权重
- [solver](solver) / [physics_modules](physics_modules) → 方程残差来源（可作为 `physics loss` 的实现参考）
- [track_geometry](track_geometry) / [infrastructure](infrastructure) → 几何、结构参数与条件输入
- [defect_injector](defect_injector) → 缺陷场参数化输入（不平顺、沉降、扣件缺失等）
- [results](results) → 标签数据（位移、加速度、轮轨力等）

建议新增一个实验目录（示例）：

- `physicsnemo_bridge/`
  - `datasets/`
  - `models/`
  - `losses/`
  - `train/`
  - `configs/`

---

## 4. 第一步：先把“输入/输出张量协议”钉死

无论 PINN 还是 PINO，先定义 I/O 协议。

### 4.1 推荐的输入变量

最小版可用：

- 轨道参数场：`k_track(x), c_track(x), m_track(x)`
- 车辆参数向量：`m_car, k_susp, c_susp, ...`
- 工况：`v`（速度）、载荷等级
- 缺陷参数：`d(x)`（局部沉降、波磨、缺失扣件等）

### 4.2 推荐的输出变量

按你关注指标选：

- `u_rail(x,t)`：轨道位移场
- `a_car(t)`：车体/构架加速度
- `F_wr(x,t)`：轮轨力
- `YQ(t)`：脱轨系数等安全指标

### 4.3 统一张量形状建议

规则时空网格任务（更偏 PINO/FNO）：

$$
X: [B, C_{in}, N_x, N_t], \quad Y:[B, C_{out}, N_x, N_t]
$$

点采样任务（更偏 PINN/MLP）：

$$
X: [N, d_{coord}+d_{param}], \quad Y:[N, d_{field}]
$$

---

## 5. 第二步：按问题类型选模型路线

## 路线 A：PINO（建议优先）

适用条件：

- 你已有大量仿真结果（比如 [results](results)）
- 目标是“输入场/参数 -> 输出场/时程”

模型建议：

- FNO / 时空 FNO / Transolver

损失建议：

$$
\mathcal{L}_{total}=\mathcal{L}_{MSE}(Y_{pred},Y_{true})+\lambda_{dyn}\|R_{dyn}\|_1
$$

其中 `R_dyn` 可来自离散动力学残差，例如：

$$
R=M\ddot{q}+C\dot{q}+Kq-f_{ext}
$$

## 路线 B：PINN

适用条件：

- 标签数据稀缺
- 更希望用方程主导训练

模型建议：

- MLP（坐标/时空点输入）

损失建议：

$$
\mathcal{L}_{total}=\lambda_f\|R_{dyn}\|_2^2+\lambda_{bc}\|R_{bc}\|_2^2+\lambda_{ic}\|R_{ic}\|_2^2
$$

---

## 6. 第三步：定义物理残差（最关键）

你可以先从最小单元做起：

### 6.1 单自由度/少自由度残差

$$
m\ddot{z}+c\dot{z}+kz=f(t)
$$

### 6.2 车辆-轨道耦合离散残差

$$
R(q,\dot q,\ddot q)=M(q)\ddot q+C(q,\dot q)\dot q+K(q)q-F_{wr}(q,\dot q)-F_{ext}
$$

将 `R` 作为 `physics loss`：

$$
\mathcal{L}_{physics}=\mathrm{mean}(|R|)
$$

建议先做“可微近似版本”，再逐步替换为更完整模型。

---

## 7. 第四步：最小可运行实验（MVP）

建议按这个顺序落地：

1. **MVP-1（纯数据）**：仅 `L_data`，先把网络训通
2. **MVP-2（弱物理）**：加小权重 `L_physics`
3. **MVP-3（完整约束）**：加 `L_constraint`（边界、初值、极值限制）

每个阶段都记录：

- 收敛曲线
- 关键工程指标误差（峰值加速度、轮轨力极值、频域主峰位置）

---

## 8. 推荐的训练配置模板（可直接抄）

```yaml
training:
  batch_size: 4
  max_epochs: 200
  lr: 1e-3

loss:
  w_data: 1.0
  w_physics: 0.01
  w_constraint: 0.1

model:
  type: fno
  in_channels: 6
  out_channels: 3
  latent_channels: 32
  num_layers: 4
```

权重调参经验：

- 先让 `w_data` 主导，保证拟合可行
- 再逐步提高 `w_physics`
- 若训练震荡，降低 `w_physics` 或对残差做归一化

---

## 9. 评价指标建议（工程导向）

除了普通 MSE，建议同时看：

1. 峰值误差：

$$
e_{peak}=\frac{|y^{pred}_{max}-y^{true}_{max}|}{|y^{true}_{max}|+\epsilon}
$$

2. 频谱一致性（主频偏差）
3. 安全指标误差（如 `Y/Q`）
4. 物理残差均值/分位数

---

## 10. 常见失败模式与修复

### 10.1 物理项一加就不收敛

原因常见是量纲不一致。先做：

- 各项损失标准化
- 分量级调权重

### 10.2 数据拟合很好但物理残差很大

说明网络在“记数据”，未学到动力学一致性。可：

- 提高 `w_physics`
- 增加工况覆盖
- 加强边界/初值约束

### 10.3 物理残差很小但指标很差

常见是方程简化过度或观测映射不完整。可：

- 补充关键力项（轮轨接触、非线性阻尼）
- 增加与指标直接相关的监督项

---

## 11. 你可以直接使用的“迁移清单”

执行顺序建议：

1. 定义 I/O 协议（变量名、单位、张量维度）
2. 选路线（PINO 优先）
3. 做最小数据管道
4. 先跑纯数据基线
5. 接入最小动力学残差
6. 做损失量纲归一化
7. 逐步加入完整约束
8. 用工程指标而不是只看 MSE

---

## 12. 与你当前三份学习文档的关系

建议按下面顺序阅读：

1. [PhysicsNeMo_物理神经网络结构学习指南.md](PhysicsNeMo_物理神经网络结构学习指南.md)
2. [darcy_physics_informed_fno_逐段源码讲解.md](darcy_physics_informed_fno_逐段源码讲解.md)
3. [ldc_pinns_train_逐段源码讲解.md](ldc_pinns_train_逐段源码讲解.md)
4. 当前文档（迁移模板）

前三份帮你“看懂 PhysicsNeMo”，本文件帮你“开始做自己的任务”。

---

## 13. 一句话结论

对于你当前轨道车辆项目，最现实路径是：

> 先用 PINO（FNO + 数据监督）建立可用基线，再逐步注入动力学残差，最后扩展到更完整的 physics-informed 系统。

---

## 14. 在当前 PINO 框架下加入 LNN（Lagrangian Neural Network）

你当前工程里真实可改的训练入口是 [pino_model/train_forward_minimal.py](pino_model/train_forward_minimal.py)，模型定义在 [pino_model/pino_architecture.py](pino_model/pino_architecture.py)。

现状（已具备）：

- 有数据损失：`loss_time`、`loss_spec`
- 有物理损失：`loss_kin`、`loss_dyn`
- 总损失已在 [pino_model/train_forward_minimal.py](pino_model/train_forward_minimal.py#L716) 组合

这意味着你不需要推翻框架，只需新增一个 `LNN` 分支并接入 loss。

### 14.1 推荐架构：PINO 主干 + LNN 物理头（最稳妥）

建议不要一上来用 LNN 替代 PINO，而是采用“双头”结构：

1. `PINOResidualHead` 继续预测状态增量 `dz`
2. 新增 `LNNHead` 学习标量拉格朗日量 `L(q, \dot q, c)`
3. 由 `LNNHead` 通过自动微分生成动力学残差

总损失改为：

$$
\mathcal{L}_{total}=\lambda_t\mathcal{L}_{time}+\lambda_s\mathcal{L}_{spec}+\lambda_k\mathcal{L}_{kin}+\lambda_d\mathcal{L}_{dyn}+\lambda_{lnn}\mathcal{L}_{lnn}
$$

其中 `L_lnn` 是基于欧拉-拉格朗日方程的残差。

---

## 15. LNN 在轨道车辆问题里的建议公式

对每个时间步，定义：

- 广义坐标 `q`
- 广义速度 `qdot`
- 条件输入 `c`（速度、轨道参数、缺陷参数等）

学习一个标量：

$$
L_\theta(q,\dot q,c)=T_\theta(q,\dot q,c)-V_\theta(q,\dot q,c)
$$

考虑阻尼/非保守力时，推荐残差写法：

$$
R_{lnn}=\frac{d}{dt}\left(\frac{\partial L_\theta}{\partial \dot q}\right)-\frac{\partial L_\theta}{\partial q}-Q_{nc}(q,\dot q,c)
$$

对应损失：

$$
\mathcal{L}_{lnn}=\mathrm{mean}\left(\|R_{lnn}\|_2^2\right)
$$

`Q_nc` 可先用你现有动力学链路近似（例如当前 `loss_dyn` 的离散力残差逻辑），后续再逐步精细化。

---

## 16. 代码落点（按你当前代码组织）

### 16.1 新增 LNN 模块文件

建议新增：

- `pino_model/lnn_head.py`

最小接口建议：

```python
class LNNHead(nn.Module):
  def forward(self, q, qdot, context=None):
    # 返回每个样本每个时间步的标量 L
    return L

def lnn_residual_loss(q, qdot, qddot, context, lnn_head):
  # autograd 计算 d/dt(dL/dqdot)-dL/dq，并与 Q_nc 比较
  return loss_lnn
```

### 16.2 训练脚本新增导入

在 [pino_model/train_forward_minimal.py](pino_model/train_forward_minimal.py#L23) 附近，新增对 `LNNHead` 的导入。

### 16.3 模型初始化位置新增 LNN

在 [pino_model/train_forward_minimal.py](pino_model/train_forward_minimal.py#L680) 当前 `PINOResidualHead` 初始化后，新增：

- `lnn_head = LNNHead(...).to(device)`

并把 `lnn_head.parameters()` 并入优化器参数。

### 16.4 训练循环中计算 `loss_lnn`

在 [pino_model/train_forward_minimal.py](pino_model/train_forward_minimal.py#L700-L716) 的 loss 计算段中：

1. 从 `z_pred` 按 `component_segments` 切出 `disp/vel/acc`
2. 映射到 `q/qdot/qddot`
3. 计算 `loss_lnn`
4. 并入总损失

建议替换为：

```python
loss = (
  args.lambda_time * loss_time
  + args.lambda_spec * loss_spec
  + args.lambda_phy * loss_kin
  + args.lambda_dyn * loss_dyn
  + args.lambda_lnn * loss_lnn
)
```

### 16.5 参数新增

在 [pino_model/train_forward_minimal.py](pino_model/train_forward_minimal.py#L769-L812) 的参数区新增：

- `--lambda-lnn`（默认建议 `1e-3`）
- `--lnn-hidden`、`--lnn-layers`
- `--lnn-warmup-epochs`（先不用 LNN，稳定后再开）

---

## 17. 一个可执行的训练策略（强烈推荐）

为避免“上来就崩”，推荐三阶段：

### Stage A：仅 PINO 基线

- `lambda_lnn = 0`
- 训练到 `loss_time/loss_dyn` 稳定

### Stage B：小权重启用 LNN

- `lambda_lnn = 1e-4 ~ 1e-3`
- 观察 `loss_lnn` 是否下降、总损失是否震荡

### Stage C：逐步提高 LNN 权重

- 每 N 个 epoch 增加 `lambda_lnn`
- 同时监控工程指标（峰值轮轨力、加速度 RMS、频域主峰）

---

## 18. LNN 接入时最容易踩的坑

### 18.1 自动微分图断裂

LNN 需要 `q`、`qdot` 可导，确保：

- 中间变量不被 `detach()`
- 未转回 `numpy`

### 18.2 量纲不一致

你当前脚本中 `z_pred` 常在归一化空间，LNN 建议在物理量空间计算：

- 先按 `z_sigma` 做尺度恢复
- 再算拉格朗日残差

### 18.3 阻尼/非保守力遗漏

轨道车辆系统强阻尼，纯保守 `L=T-V` 常不够。必须加入 `Q_nc` 或等价耗散项，否则会出现“残差小但预测偏”的情况。

### 18.4 全 35 自由度直接上 LNN 难收敛

建议先从子系统开始：

1. 车体垂向 + 俯仰（2~3 DOF）
2. 再扩展到转向架
3. 最后并到全 35 DOF

---

## 19. 最小改造清单（按顺序执行）

1. 新增 `pino_model/lnn_head.py`
2. 在训练脚本加入 `lambda_lnn` 与 `lnn_head`
3. 在 loss 汇总处并入 `loss_lnn`
4. 先跑 `lambda_lnn=0` 验证不破坏原链路
5. 再逐步打开 `lambda_lnn`

这条路径对你当前代码侵入最小、风险最低、最容易回退。

---

## 20. 相较“单独框架”的优势：正向计算与逆向评估

这里将“组合框架”定义为：

- PINO 主干（高维算子学习）
- 现有动力学残差（`loss_dyn`）
- LNN 物理结构残差（`loss_lnn`）

即：

$$
\mathcal{L}_{total}=\lambda_t\mathcal{L}_{time}+\lambda_s\mathcal{L}_{spec}+\lambda_k\mathcal{L}_{kin}+\lambda_d\mathcal{L}_{dyn}+\lambda_l\mathcal{L}_{lnn}
$$

### 20.1 正向计算（Forward）

目标：给定轨道参数/缺陷/工况，预测响应（位移、加速度、轮轨力等）。

#### 对比仅 PINO

优势：

1. 外推稳定性更好（新速度、新缺陷组合更不易漂移）
2. 长时程滚动预测更稳（物理残差抑制误差累积）
3. 峰值指标更可靠（如轮轨力峰值、车体加速度峰值）

原因：

- PINO 负责表达能力
- LNN + 动力学残差负责物理一致性

#### 对比仅 LNN/PINN

优势：

1. 高维时空响应学习更高效
2. 训练收敛更容易
3. 工程可用精度更快达到

原因：

- 仅 LNN/PINN 在高维场映射上常训练困难
- PINO 在“场到场映射”上更有优势

### 20.2 逆向评估（Inverse）

目标：根据观测响应反推参数/缺陷（刚度、阻尼、缺陷幅值与位置等）。

#### 对比仅数据驱动框架

优势：

1. 可辨识性更好（多解空间缩小）
2. 对噪声更鲁棒（不易被噪声牵引到非物理解）
3. 结果可解释性更强（满足动力学残差）

原因：

逆问题通常病态，仅用拟合项：

$$
\min \|y-\hat y\|
$$

容易出现多个参数都能拟合同一响应。加入物理约束后：

$$
\min \|y-\hat y\|+\lambda_d\|R_{dyn}\|+\lambda_l\|R_{lnn}\|
$$

可显著压缩不合理解空间。

#### 对比仅物理框架

优势：

1. 对复杂轨道缺陷激励有更强表达能力
2. 反演效率更高（借助学习到的算子近似）
3. 可直接利用已有大量仿真数据

---

## 21. 可执行的对比实验设计（建议直接落地）

为证明上述优势，建议做四组消融：

- **A组（PINO-only）**：`lambda_dyn>0, lambda_lnn=0`
- **B组（LNN-only）**：关闭 PINO 主干或仅保留 LNN 分支（小规模子系统）
- **C组（PINO + Dyn）**：`lambda_dyn>0, lambda_lnn=0`
- **D组（PINO + Dyn + LNN）**：`lambda_dyn>0, lambda_lnn>0`（目标方案）

> 实操时可将 A/C 合并为“仅现有框架”，关键是 D 与“无 LNN”对照。

### 21.1 正向任务测试集

建议三类工况：

1. **同分布测试**：与训练分布一致
2. **弱外推测试**：速度与缺陷幅值超出训练区间 10~20%
3. **强外推测试**：组合工况（速度 × 缺陷类型 × 参数扰动）

评价指标：

- `MSE / MAE`
- 峰值误差：

$$
e_{peak}=\frac{|y^{pred}_{max}-y^{true}_{max}|}{|y^{true}_{max}|+\epsilon}
$$

- 频谱主峰偏差
- `R_dyn`、`R_lnn` 分位数统计（P50/P90/P99）

### 21.2 逆向任务测试集

建议两类反演：

1. **参数反演**：刚度、阻尼、车速
2. **缺陷反演**：缺陷位置、幅值、波长（或等效参数）

评价指标：

- 参数相对误差
- 置信区间宽度（多次噪声重采样）
- 重建响应误差
- 物理残差一致性（反演解是否满足动力学约束）

### 21.3 判据（建议）

当 D 组满足以下条件，即可认为“组合框架成立”：

1. 正向外推集上峰值误差显著低于无 LNN 组
2. 逆向反演误差与方差同时下降
3. 物理残差统计量（尤其 P90/P99）显著改善

---

## 22. 成本与收益的工程化结论

### 成本

- 训练复杂度上升（多损失权重与 warmup）
- 自动微分计算图更重（显存与耗时增加）
- 需要更谨慎的量纲归一化

### 收益

- 正向预测在外推场景下更稳
- 逆向评估更可辨识、更可解释
- 更适合“正向-逆向一体化”工程流程

对你的轨道车辆场景，若目标包含 **跨工况鲁棒预测 + 参数/缺陷反演**，组合框架通常优于任何单一框架。
