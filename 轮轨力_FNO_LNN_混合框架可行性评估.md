# 轮轨力 FNO + 振动方程 LNN 混合框架可行性评估

## 1. 问题定义

目标映射为：

$$
\text{初始自由度状态} + \text{不平顺激励} \rightarrow \text{车体振动响应}
$$

更具体地，可写为：

$$
(q_0, \dot{q}_0, r(t), p) \rightarrow y(t)
$$

其中：

- $q_0, \dot{q}_0$：系统初始位移与速度
- $r(t)$：轮轨不平顺激励
- $p$：车辆、悬挂、轨道等参数
- $y(t)$：车体、构架、轮对的振动响应

当前希望在该映射中嵌入车辆-轨道耦合动力学物理信息，但会遇到一个核心矛盾：

1. 若显式引入轮轨力，则必须面对高度非线性的接触建模、查表、接触状态切换等复杂问题；
2. 若完全绕开轮轨力，则动力学主链路会中断，难以保证模型具有真实物理可解释性。

因此，本问题本质上不是“能否拟合响应”，而是“如何在可训练前提下，保留轮轨接触到车体响应的物理传递链路”。

---

## 2. 总体可行性判断

结论：**该框架可行，但不建议首版采用纯黑箱端到端方式直接实现完整轮轨接触 + 全耦合动力学。**

更合理的路线是构建一个**结构化混合模型**：

- 轮轨接触部分：采用“解析物理主干 + 可学习修正”
- 动力学传播部分：采用显式二阶动力学结构或带耗散项的 LNN
- 输出部分：预测车体、构架、轮对垂向响应

推荐的总体表达形式为：

$$
F_{wr} = F_{phys} + \Delta F_{\theta}
$$

再将其代入结构化动力学系统：

$$
M(q)\ddot{q} + C(q, \dot{q})\dot{q} + K(q)q = B_f F_{wr} + B_r r(t)
$$

这种方式的优点是：

- 保留轮轨力这一关键中间物理量；
- 避免将全部困难都交给神经网络；
- 有利于训练稳定性与结果可解释性；
- 便于从简化垂向模型逐步扩展到更完整的多自由度耦合系统。

---

## 3. 为什么不建议首版直接做“纯 FNO + 纯 LNN”

### 3.1 轮轨力误差会被动力学积分放大

轮轨接触力处于激励链路前端，一旦该模块存在系统性偏差，误差会沿着：

$$
F_{wr} \rightarrow \ddot{q} \rightarrow \dot{q} \rightarrow q
$$

逐步积累并被放大。也就是说，后端动力学模块越“物理严格”，越会把前端接触力误差真实地传播出来。

因此，轮轨力模块不能只追求拟合精度，更要重视物理一致性。

### 3.2 纯 LNN 不完全适合当前系统

标准 LNN 更适合近保守系统，而当前车辆-轨道系统具有以下特征：

- 阻尼项明显存在；
- 外部激励显著存在；
- 轮轨接触本身是非保守且可能状态切换的；
- 若引入蠕滑力，则还存在明显耗散性。

因此，更适合的不是最原始的 LNN，而是：

- 带 Rayleigh 耗散项的 Lagrangian 网络；
- 结构化二阶动力学网络；
- 或者“解析动力学主干 + 神经网络学习未知项”的混合方案。

### 3.3 FNO 未必是轮轨力模块的首选

FNO 更擅长学习：

- 场到场映射；
- 序列到序列算子；
- 高维输入输出上的全局卷积结构。

如果轮轨接触模块只做局部本构关系学习，例如：

$$
(\delta, \dot{\delta}, \xi_x, \xi_y, \text{几何参数}, \text{材料参数}) \rightarrow (F_n, F_x, F_y)
$$

那么这本质上更接近一个低维非线性本构映射，此时：

- MLP
- ResMLP
- DeepONet
- 小型 Transformer

通常会比 FNO 更节省数据、更容易训练、更稳定。

若你的接触模块要学习的是：

$$
\text{整段时序状态} \rightarrow \text{整段轮轨力时序}
$$

则 FNO 才更有明显优势。

---

## 4. 赫兹接触还是 Kalker 接触

### 4.1 若当前任务以垂向振动响应为主

若当前目标仍然是：

- 车体垂向振动响应；
- 构架垂向响应；
- 轮对垂向振动；
- 以不平顺激励引起的垂向传递为主；

那么建议**首版优先使用赫兹法向接触模型**。

原因如下：

1. 赫兹模型直接对应法向接触主通道；
2. 对垂向动力响应而言，法向接触是最关键的物理量；
3. 数学形式清晰，便于微分、约束和训练；
4. 可用作强物理先验主干；
5. 能作为后续复杂接触模型的基线。

法向赫兹接触可以写为：

$$
F_n = C_H \delta_+^{3/2}
$$

其中 $\delta_+ = \max(\delta, 0)$。

### 4.2 若未来扩展到横向/蠕滑/蛇行/安全指标

如果后续任务扩展到：

- 横向动力学；
- 蠕滑力；
- 脱轨系数；
- 轮重减载；
- 蛇行稳定性；

则必须逐步引入更强的接触物理约束，例如：

- 线性 Kalker 理论；
- FASTSIM；
- 更复杂的非线性切向接触方法。

此时接触模块不再只是法向压入，而是需要同时描述：

- 法向接触；
- 切向蠕滑；
- 摩擦约束；
- 接触斑椭圆变化；
- 接触点位置和局部几何变化。

### 4.3 推荐结论

- **首版：赫兹法向接触 + 可学习修正项**
- **后续增强：Kalker/FASTSIM 约束下的切向接触扩展**

因此，不建议一开始就把复杂 Kalker 接触完整塞进网络训练主流程中。

---

## 5. 推荐的首版混合架构

### 5.1 输入输出设计

#### 输入

- 初始状态：$q_0, \dot{q}_0$
- 不平顺激励：$r(t)$
- 参数向量：$p$

在张量层面可定义为：

- `init_state`: `[B, n_state]`
- `irr`: `[B, 4, T]`
- `params`: `[B, n_param]`

#### 输出

建议输出与你当前数据集一致的 21 通道：

- 位移：7 通道
- 速度：7 通道
- 加速度：7 通道

即：

- `resp`: `[B, 21, T]`

这样可以直接与当前数据集和现有简化动力学方程对接。

---

### 5.2 模块分层

#### 模块 A：初始状态编码器

作用：将 $q_0, \dot{q}_0, p$ 编码为初始隐状态。

可选结构：

- MLP
- 小型时序编码器
- 参数条件化编码层

#### 模块 B：轮轨接触模块

作用：根据轮对局部状态和不平顺激励，预测轮轨接触力。

推荐形式：

$$
F_n(t) = F_n^{Hertz}(t) + \Delta F_n^{\theta}(t)
$$

输入建议包括：

- 轮对位移/速度局部量；
- 不平顺输入；
- 参数向量；
- 必要时加入轨道局部几何特征。

若首版仅垂向，可只预测法向力或等效垂向接触力。

#### 模块 C：动力学传播层

作用：将接触力作为外部激励，输入结构化动力学方程并传播为系统响应。

建议采用以下两种之一：

1. **显式动力学积分器**
   - 优点：物理清晰、稳定性可控、解释性强。
2. **结构化二阶动力学网络 / 带耗散项 LNN**
   - 优点：保留物理结构、可学习未知部分。

不建议首版将动力学完全交给无结构黑箱网络。

#### 模块 D：响应输出层

输出：

- 车体垂向位移/速度/加速度
- 构架垂向位移/速度/加速度
- 轮对垂向位移/速度/加速度

---

## 6. 物理损失函数设计

总损失建议为：

$$
\mathcal{L} = \lambda_{resp}\mathcal{L}_{resp} + \lambda_{dyn}\mathcal{L}_{dyn} + \lambda_{contact}\mathcal{L}_{contact} + \lambda_{state}\mathcal{L}_{state}
$$

### 6.1 响应监督损失

$$
\mathcal{L}_{resp} = \|y_{pred} - y_{true}\|^2
$$

建议对以下量加监督：

- 车体响应
- 构架响应
- 轮对响应

如果条件允许，也可加入频域损失，以保证主频特征匹配。

### 6.2 动力学残差损失

将预测响应代入结构化动力学方程：

$$
M\ddot{q} + C\dot{q} + Kq - B_fF_n - B_rr = 0
$$

对应损失：

$$
\mathcal{L}_{dyn} = \|M\ddot{q} + C\dot{q} + Kq - B_fF_n - B_rr\|^2
$$

对于你当前简化垂向模型，这部分已经具备良好的实现基础。

### 6.3 接触物理损失

首版建议至少加入以下约束：

#### (1) 法向力非负

$$
F_n \ge 0
$$

#### (2) 无压入不接触

$$
\delta \le 0 \Rightarrow F_n \approx 0
$$

#### (3) 接近赫兹主律

$$
F_n \approx C_H \delta_+^{3/2}
$$

#### (4) 单调性约束

$$
\frac{\partial F_n}{\partial \delta} \ge 0
$$

### 6.4 中间状态监督

如果可以从仿真中导出中间量，建议额外监督：

- 轮轨力
- 轮对响应
- 构架响应
- 压入量/接触状态

这一步非常关键，因为如果只监督车体响应，轮轨力映射会出现“不可辨识性”问题，即不同接触力轨迹可能产生相似的车体响应。

---

## 7. 关于“轮轨力不可绕过”的判断

这一点判断是正确的。

在车体响应形成链路中，轮轨力处于关键传递位置：

$$
\text{不平顺} \rightarrow \text{轮轨接触} \rightarrow \text{轮对受力} \rightarrow \text{构架响应} \rightarrow \text{车体响应}
$$

如果完全绕过轮轨力，模型虽然仍可能学到输入到输出的统计映射，但会出现以下问题：

1. 动力学链路不清楚；
2. 模型外推能力弱；
3. 难以推广到新速度、新轨道、新参数；
4. 不能解释哪些接触条件引起了哪些响应；
5. 物理约束难以正确施加。

因此，轮轨力不能彻底省略，但可以采用“解析主干 + 学习修正”的方式降低建模复杂度。

---

## 8. 推荐的开发路线

### 第一阶段：纯垂向首版

目标：快速构建稳定可训练基线。

建议配置：

- 输入：初始状态 + 垂向不平顺 + 参数
- 接触：赫兹法向接触
- 动力学：简化垂向动力学方程
- 输出：21 通道垂向响应
- 损失：响应损失 + 动力学残差 + 赫兹接触约束

该阶段重点不在“最复杂”，而在“先打通物理链路”。

### 第二阶段：接触修正学习

目标：让神经网络学习赫兹模型难以覆盖的复杂接触效应。

建议形式：

$$
F_n = F_n^{Hertz} + \Delta F_n^{\theta}
$$

其中 $\Delta F_n^{\theta}$ 的规模应受约束，避免网络完全推翻物理主干。

### 第三阶段：扩展切向接触

目标：增强横向/蠕滑相关真实性。

建议方式：

- 小蠕滑区先引入线性 Kalker 约束；
- 后续再逐步过渡到 FASTSIM 或更复杂接触模型；
- 神经网络只学习难以解析建模的修正部分。

### 第四阶段：更完整耦合系统

在首版稳定之后，再逐步引入：

- 横向自由度
- 摇头/滚摆自由度
- 更复杂轨道几何与轮轨接触状态
- 多物理量联合预测

---

## 9. 最终建议

综合考虑当前任务目标、现有数据结构、物理复杂度和训练稳定性，建议如下：

### 9.1 建议采用的首版方案

- **动力学主干**：结构化二阶动力学方程 / 带耗散项 LNN
- **接触主干**：赫兹法向接触
- **学习模块**：学习赫兹接触修正项，而不是从零学习全部轮轨力
- **训练目标**：车体 + 构架 + 轮对垂向响应联合监督
- **物理损失**：动力学残差 + 接触约束 + 中间状态监督

### 9.2 不建议的首版方案

- 直接使用完整 Kalker 接触 + FNO + LNN 做全端到端训练；
- 在只有车体响应监督的情况下，让网络自由学习全部轮轨力；
- 将动力学和接触全部交给黑箱模型。

### 9.3 一句话判断

**该框架是可行的，但最佳实现路线不是“复杂模型一步到位”，而是“赫兹接触打底、结构化动力学传播、神经网络学习修正项、再逐步增强接触复杂度”。**

---

## 10. 建议的下一步工作

建议下一步直接落到工程实现层面，明确以下内容：

1. 输入张量与输出张量的最终定义；
2. 动力学层是采用显式积分器还是带耗散项 LNN；
3. 轮轨力模块是采用 FNO、MLP 还是其他算子网络；
4. 训练损失中各项权重如何设置；
5. 当前数据集中是否能导出轮轨力或压入量等中间监督量。

若以工程落地优先，推荐先做：

- `Hertz + 动力学残差 + 21通道响应监督`

待首版训练稳定后，再扩展到：

- `Kalker/FASTSIM + 更完整耦合自由度 + 更强物理约束`

---

## 11. 网络结构图（工程实现版）

可按如下流程实现：

```text
init_state(q0,dq0), irr(t), params(p)
                  │
                  ├── Encoder ───────────────┐
                  │                          │
                  └── Wheel Local Features ──┴─> Contact Net
                                                             (Hertz + ΔFθ)
                                                                      │
                                                                      ▼
                                                         F_wr(t) / F_n(t)
                                                                      │
                                                                      ▼
                                                   Dynamics Integrator
                                       Mqdd + Cqd + Kq = BfFwr + Brr
                                                                      │
                                                                      ▼
                                                   Response Head / Direct Readout
                                                                      │
                                                                      ▼
                                                    y_pred = [disp, vel, acc] (21ch)
```

建议首版采用“先算接触，再做动力学传播”的串联模式，不建议首版做全黑箱并联端到端。

---

## 12. PyTorch 模块划分草案

以下为推荐文件组织（示例）：

- `models/encoder.py`
   - `StateParamEncoder`
- `models/contact_hybrid.py`
   - `HertzContactLayer`
   - `ContactResidualNet`（可选 MLP / FNO）
   - `HybridContactModel`
- `models/dynamics_layer.py`
   - `StructuredDynamicsLayer`
   - `TimeIntegrator`（Euler / RK / Newmark 近似可微版本）
- `models/response_head.py`
   - `ResponseHead`
- `models/system_model.py`
   - `VehicleTrackHybridModel`

### 12.1 核心接口建议

```python
class VehicleTrackHybridModel(nn.Module):
      def forward(self, init_state, irr, params):
            # init_state: [B, n_state]
            # irr:        [B, 4, T]
            # params:     [B, n_param]
            # return:
            # y_pred:     [B, 21, T]
            # aux:        dict(F_wr=..., q=..., dq=..., ddq=..., delta=...)
            ...
```

### 12.2 接触模块建议

```python
class HybridContactModel(nn.Module):
      def forward(self, wheel_feat, params):
            # Hertz 主干
            Fn_h = hertz_fn(delta=wheel_feat[..., 0], ch=params[..., 0])

            # 学习修正
            dFn = self.res_net(wheel_feat, params)

            # 混合输出
            Fn = Fn_h + dFn
            Fn = torch.clamp(Fn, min=0.0)
            return Fn, {"Fn_h": Fn_h, "dFn": dFn}
```

### 12.3 动力学层建议

```python
class StructuredDynamicsLayer(nn.Module):
      def forward(self, q, dq, F_wr, irr, params):
            # 构造 M, C, K 或其简化参数化
            # 计算 ddq = M^{-1}(Bf F_wr + Br irr - C dq - K q)
            return ddq
```

---

## 13. 训练损失伪代码（首版）

```python
for batch in loader:
      init_state, irr, params, y_true = batch

      y_pred, aux = model(init_state, irr, params)
      q, dq, ddq = aux["q"], aux["dq"], aux["ddq"]
      F_wr = aux["F_wr"]
      delta = aux.get("delta", None)

      # 1) 响应监督
      L_resp = mse(y_pred, y_true)

      # 2) 动力学残差
      # R = M ddq + C dq + K q - Bf F_wr - Br irr
      R_dyn = dynamics_residual(q, dq, ddq, F_wr, irr, params)
      L_dyn = (R_dyn ** 2).mean()

      # 3) 接触物理约束（示例）
      L_nonneg = relu(-F_wr).mean()                 # F_wr >= 0
      L_hertz = hertz_consistency(F_wr, delta, params)
      L_contact = L_nonneg + L_hertz

      # 4) 可选中间状态监督
      L_state = 0.0
      if "F_wr_true" in batch:
            L_state = mse(F_wr, batch["F_wr_true"])

      loss = (
            w_resp * L_resp
            + w_dyn * L_dyn
            + w_contact * L_contact
            + w_state * L_state
      )

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

### 13.1 初始权重建议（经验起点）

- $w_{resp}=1.0$
- $w_{dyn}=0.1$
- $w_{contact}=0.1$
- $w_{state}=0.5$（若有轮轨力标签）

训练早期建议先以 `L_resp` 主导，待响应收敛后逐步提高 `w_dyn` 与 `w_contact`。

### 13.2 验证指标建议

- 时域：RMSE / MAE（车体、构架、轮对分开统计）
- 频域：主峰频率误差、谱能量分布误差
- 物理一致性：
   - 动力学残差分位数（median / p95）
   - 接触力非负率
   - 赫兹一致性误差

---

以上三节可直接作为首版实现蓝图：

1. 先完成模块接口；
2. 用当前 21 通道数据完成闭环训练；
3. 在验证通过后再扩展切向接触与更高维自由度。
