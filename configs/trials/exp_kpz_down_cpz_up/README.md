# 试验参数目录：exp_kpz_down_cpz_up

说明：
- 本目录用于做“降低一次垂向刚度 Kpz + 提高一次垂向阻尼 Cpz”的频域抑制试验。
- 运行时通过参数指定：`--param_profile_dir configs/trials/exp_kpz_down_cpz_up`
- 未提供的 YAML 文件会自动回退到 `configs/standard`。

本试验仅覆盖：
- `vehicle_params.yaml`（高速客车的一系垂向参数）
