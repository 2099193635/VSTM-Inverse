<!--
 * @Author: 2099193635 2099193635@qq.com
 * @Date: 2026-04-09 15:11:55
 * @LastEditors: 2099193635 2099193635@qq.com
 * @LastEditTime: 2026-04-09 15:14:29
 * @FilePath: /VTCM_PYTHON/inverse_model/README.md
 * @Description: 
 * 
 * Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
-->
## Tensor Dimension Convention

B = batch_size  
T = time_steps (传感器观测序列长度，来自 WindowConfig.window_size) 
n_s = n_sensors（传感器通道数：默认 3 = 车体Z + 前构架Z + 后构架Z）  
n_c = n_cond（条件维度：40 vehicle_params + 1 vx_mps = 41）  
L = spatial_len（空间域查询点数，默认 = T，通过 x = v·t 对齐）  
n_d = n_directions（不平顺方向数：1=垂向 or 2=垂+横）  
p = branch_modes（DeepONet 系数数，默认 = 32）  
w = width（隐通道宽度，默认 = 64，与 PINOConfig 一致）
