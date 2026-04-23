# 验证与可视化功能说明

## 功能概述

在训练过程中，每`val_every`个epoch会自动执行以下操作：

1. **验证集评估** - 在验证集上运行完整前向推理
2. **预测可视化** - 生成预测值与真实值的对比曲线图
3. **文件保存** - 所有可视化图保存在 `checkpoints/predictions/` 目录

## 使用方法

### 基本用法（推荐）

```bash
python inverse_model/train.py \
  --epochs 50 \
  --batch_size 8 \
  --val_every 5 \
  --log_every 5
```

**参数说明：**
- `--epochs 50`: 训练50个epoch
- `--batch_size 8`: 批大小为8
- `--val_every 5`: 每5个epoch进行一次验证和可视化
- `--log_every 5`: 每5个epoch打印一次日志
- `--use_full_seq`: 自动启用（使用temporal length = 1000的full_seq数据）

### 配置详细参数

```bash
python inverse_model/train.py \
  --epochs 100 \
  --batch_size 16 \
  --val_every 5 \
  --log_every 5 \
  --physics_mode "none" \
  --dataset_dir "/workspace/VTCM_PYTHON/datasets/VTCM_inverse"
```

## 输出文件结构

```
checkpoints/
├── predictions/                              # 每5步生成的可视化图
│   ├── pred_vs_true_epoch_0005.png
│   ├── pred_vs_true_epoch_0010.png
│   ├── pred_vs_true_epoch_0015.png
│   └── pred_vs_true_epoch_0020.png
├── checkpoint_epoch_0020.pt                  # 模型检查点（每20步保存）
├── checkpoint_latest.pt                      # 最新模型
├── train_history.npz                         # 训练历史（所有loss）
└── loss_curve.png                            # Loss曲线总图
```

## 可视化图说明

### 图像内容

每个 `pred_vs_true_epoch_XXXX.png` 文件包含：

- **4个子图**：对应验证集中的4个样本
- **蓝色实线**：Ground Truth（真实的轨道不平顺）
- **红色虚线**：预测值（模型推理结果）
- **标题**：样本号和RMSE值（衡量预测精度）

### 图像尺寸

- 宽度：1389像素
- 高度：788像素
- 分辨率：100 DPI

### 解读指标

- **RMSE < 0.001**：优秀（预测与真实值接近）
- **RMSE 0.001-0.005**：良好
- **RMSE > 0.005**：需要改进

## 数据集说明

### Full-Seq模式（默认使用）

- **temporal_length**: 1000个空间点（整个轨道不平顺序列）
- **数据量**：
  - 训练集：160个样本（160 files）
  - 验证集：20个样本（20 files）
  - 测试集：28个样本（28 files）
- **存储**：
  - `train_full_seq.hdf5`: 2.2 MB
  - `validation_full_seq.hdf5`: 319 KB
  - `test_full_seq.hdf5`: 320 KB

### 数据特征

- **输入加速度**：[B, 1000, 3] （vertical, lateral, roll）
- **输出不平顺**：[B, 1000, 1] （垂直方向）
- **条件参数**：[B, 17] （自然频率、阻尼比等）

## 代码修改概览

### 1. inverse_trainer.py 的修改

#### 新增函数：`plot_predictions()`
- 绘制预测vs真实对比曲线
- 支持多个样本并行展示
- 自动计算RMSE并显示在图标题中

#### 新增方法：`_visualize_predictions()`
- 在验证集上进行前向推理
- 调用 `plot_predictions()` 生成图像
- 在每个 `val_every` 间隔执行

#### 修改方法：`fit()`
- 新增参数 `val_every` （默认=5）
- 改进验证逻辑：仅在指定间隔验证
- 自动创建 `predictions/` 目录
- 在验证后调用可视化函数

### 2. train.py 的修改

#### 参数更新
- `--use_full_seq`: 改为 `default=True`（自动启用）
- 新增 `--val_every`：控制验证和可视化频率（默认=5）

#### 调用更新
```python
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=args.epochs,
    ckpt_dir=ckpt_dir,
    log_every=args.log_every,
    val_every=args.val_every,  # 新增
)
```

## 训练运行示例

### 示例命令
```bash
python inverse_model/train.py --epochs 20 --batch_size 8 --val_every 5
```

### 输出日志
```
14:00:51 [INFO] train: Device: cuda
14:00:51 [INFO] train: Loading train dataset: .../train_full_seq.hdf5
14:00:51 [INFO] train: Config: epochs=20, batch=8, physics=none, full_seq=True
14:00:51 [INFO] train: Train samples: 160, Val samples: 20
14:00:53 [INFO] train: Training PCNIO | 20 epochs | cuda
14:00:57 [INFO] inverse_trainer: Epoch    5 | train 1.0573 ... | val 1.2094
14:00:58 [INFO] inverse_trainer: Prediction plot saved: .../pred_vs_true_epoch_0005.png
14:01:01 [INFO] inverse_trainer: Epoch   10 | train 1.0632 ... | val 1.2002
14:01:02 [INFO] inverse_trainer: Prediction plot saved: .../pred_vs_true_epoch_0010.png
14:01:10 [INFO] train: Loss history saved: .../train_history.npz
14:01:10 [INFO] train: Loss curve saved: .../loss_curve.png
```

## 常见问题

### Q: 为什么只在 `val_every` 间隔进行验证？
A: 这样可以减少验证时间成本，加快训练速度。同时保留关键检查点的可视化记录。

### Q: 可以改变 `val_every` 的值吗？
A: 可以。使用 `--val_every N` 参数：
- `--val_every 1`：每个epoch都验证（最详细但最慢）
- `--val_every 10`：每10个epoch验证一次（快速但检查点少）

### Q: 图像太小看不清？
A: 可以修改 `plot_predictions()` 中的 `figsize` 参数，例如：
```python
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5*nrows))  # 增大图像
```

### Q: 如何查看所有生成的可视化图？
A: 使用文件浏览器打开：
```
/workspace/VTCM_PYTHON/inverse_model/checkpoints/predictions/
```

## 下一步建议

1. **物理模式测试**：加入 `--physics_mode frf` 或 `--physics_mode pinn`
2. **长期训练**：运行 `--epochs 100` 观察收敛趋势
3. **批量参数搜索**：测试不同的 `--val_every` 和 `--batch_size`
4. **结果分析**：使用 `train_history.npz` 绘制自定义统计图表

## 技术细节

### 数据流向

```
验证集DataLoader
    ↓
_visualize_predictions()
    ├─→ model.forward() on GPU
    ├─→ 收集所有预测结果
    └─→ plot_predictions()
         └─→ 保存PNG图像
```

### 性能影响

- **验证时间**：~0.5秒/epoch（20个样本，batch_size=8）
- **可视化时间**：~1秒/epoch（matplotlib绘图）
- **总额外时间**：每个 `val_every` 周期增加 ~5%

### 内存使用

- **峰值**：在 `_visualize_predictions()` 中累积所有预测（~5-10MB）
- **恢复**：函数结束后自动释放

