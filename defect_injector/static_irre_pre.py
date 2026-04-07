r'''
Author: Niscienc 60505912+2099193635@users.noreply.github.com
Date: 2026-03-16 15:43:05
LastEditors: Niscienc 60505912+2099193635@users.noreply.github.com
LastEditTime: 2026-03-16 15:53:56
FilePath: \VTCM_PYTHON\defect_injector\static_irre_pre.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
from __future__ import annotations

from pathlib import Path
import datetime
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_numeric_if_possible(series: pd.Series) -> pd.Series:
	"""尽量转数值；无法转换时保留原始文本。"""
	converted = pd.to_numeric(series, errors="coerce")
	# 如果全部变成 NaN，说明不是数值列，返回原列
	if converted.notna().sum() == 0:
		return series
	return converted


def _detect_jdt_encoding(file_path: str | Path, candidates: Optional[Iterable[str]] = None) -> str:
	"""检测 .jdt 文件编码，默认优先使用国标中文编码。"""
	file_path = Path(file_path)
	candidates = list(candidates or ("gb18030", "gbk", "utf-8-sig"))
	raw = file_path.read_bytes()[:4096]

	for enc in candidates:
		try:
			raw.decode(enc)
			return enc
		except UnicodeDecodeError:
			continue

	# 保底返回 gb18030（该类检测文件通常是该编码）
	return "gb18030"


def read_jdt(file_path: str | Path, encoding: Optional[str] = None) -> pd.DataFrame:
	"""
	读取 .jdt 静态检测文件。

	特征：
	1) 文本格式；
	2) 制表符分隔（\t）；
	3) 常见编码 gb18030/gbk。
	"""
	file_path = Path(file_path)
	if not file_path.exists():
		raise FileNotFoundError(f"文件不存在: {file_path}")

	enc = encoding or _detect_jdt_encoding(file_path)

	df = pd.read_csv(
		file_path,
		sep="\t",
		encoding=enc,
		engine="python",
		on_bad_lines="skip",
	)

	# 清理列名中的首尾空白
	df.columns = [str(c).strip() for c in df.columns]

	# 尽量将字段转为数值，方便后续计算
	for c in df.columns:
		df[c] = _to_numeric_if_possible(df[c])

	return df


def tidy_jdt_with_single_mileage(df: pd.DataFrame, mileage_col: Optional[str] = None) -> pd.DataFrame:
	"""
	把 .jdt 的 (X)/(Y) 成对列整理为：
	- 一列统一里程：`里程`
	- 各通道仅保留 Y 值，并去掉列名中的 `(Y)`

	优先使用 `实际里程(Y)` 作为里程；若不存在则退化为第一个可用 `(...X)` 列。
	"""
	if df.empty:
		return df.copy()

	cols = [str(c).strip() for c in df.columns]

	# 1) 选择统一里程列
	if mileage_col and mileage_col in cols:
		mileage_src = mileage_col
	elif "实际里程(Y)" in cols:
		mileage_src = "实际里程(Y)"
	elif "实际里程(X)" in cols:
		mileage_src = "实际里程(X)"
	else:
		x_cols = [c for c in cols if c.endswith("(X)")]
		if not x_cols:
			raise ValueError("未找到可用里程列：既没有 '实际里程(Y)'，也没有任何 '(X)' 列。")
		mileage_src = x_cols[0]

	result = pd.DataFrame()
	result["里程"] = _to_numeric_if_possible(df[mileage_src]).astype("float64")

	# 2) 收集 Y 列；只保留测值，统一去掉后缀 (Y)
	used_names = {"里程"}
	for c in cols:
		if not c.endswith("(Y)"):
			continue

		base = c[:-3].strip()
		if base == "实际里程":
			# 里程已单独作为统一列
			continue

		new_name = base
		# 处理重名：自动加后缀 _2, _3...
		if new_name in used_names:
			idx = 2
			while f"{new_name}_{idx}" in used_names:
				idx += 1
			new_name = f"{new_name}_{idx}"

		result[new_name] = _to_numeric_if_possible(df[c])
		used_names.add(new_name)

	# 3) 里程排序（升序），并重建索引
	result = result.sort_values("里程", kind="mergesort").reset_index(drop=True)
	return result


def export_for_irregularity_external_import(
	tidy_df: pd.DataFrame,
	out_dir: str | Path,
	file_prefix: str = "external_irre",
	use_relative_mileage: bool = False,
	channel_map: Optional[dict] = None,
) -> dict:
	"""
	把静态不平顺整理结果导出为 irregularity.py 可直接读取的 4 文件格式。

	目标文件（两列数值，无表头）：
	- VL: 左轨垂向
	- VR: 右轨垂向
	- LL: 左轨横向
	- LR: 右轨横向

	说明：
	- 第1列为里程，默认保留绝对里程（use_relative_mileage=False）；若设为 True 则转为相对里程（从 0 开始）。
	- 第2列为不平顺幅值（保持原单位，通常为 mm；由 irregularity.py 的 Factor 决定是否转 m）
	- meta 文件中写入 origin_abs（绝对里程起点），供 Irregularity 的 external_mileage_mode='absolute' 使用。
	"""
	if "里程" not in tidy_df.columns:
		raise ValueError("tidy_df 缺少 '里程' 列，无法导出外部导入文件。")

	default_map = {
		"VL": "实测左高低",
		"VR": "实测右高低",
		"LL": "实测左轨向",
		"LR": "实测右轨向",
	}
	if channel_map:
		default_map.update(channel_map)

	missing = [name for name in default_map.values() if name not in tidy_df.columns]
	if missing:
		raise ValueError(f"缺少导出所需列: {missing}")

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	x = pd.to_numeric(tidy_df["里程"], errors="coerce").to_numpy(dtype=float)
	valid_x = np.isfinite(x)
	if valid_x.sum() < 2:
		raise ValueError("里程有效点不足，无法导出。")

	origin_abs = float(np.nanmin(x))  # 绝对里程起点，始终记录
	if use_relative_mileage:
		x = x - origin_abs

	paths = {}
	for key, col in default_map.items():
		y = pd.to_numeric(tidy_df[col], errors="coerce").to_numpy(dtype=float)
		valid = np.isfinite(x) & np.isfinite(y)
		xy = np.column_stack([x[valid], y[valid]])

		# 按里程升序
		order = np.argsort(xy[:, 0])
		xy = xy[order]

		out_file = out_dir / f"{file_prefix}_{key}.txt"
		np.savetxt(out_file, xy, fmt="%.10f")
		paths[key] = str(out_file)

	# 额外保存一个路径说明，便于直接喂给 irregularity.py 的 external_files
	mileage_mode = "relative" if use_relative_mileage else "absolute"
	meta_file = out_dir / f"{file_prefix}_external_files.txt"
	with meta_file.open("w", encoding="utf-8") as f:
		f.write("# 可直接用于 Irregularity.excitation_irregularity(external_files=...)\n")
		f.write(f"# mileage_mode={mileage_mode}\n")
		f.write(f"# origin_abs={origin_abs:.6f}  (绝对里程起点，单位与数据相同)\n")
		for k in ["VL", "VR", "LL", "LR"]:
			f.write(f"{k}={paths[k]}\n")
		f.write(f"origin_abs={origin_abs:.6f}\n")
		f.write(f"mileage_mode={mileage_mode}\n")
	paths["META"] = str(meta_file)
	paths["origin_abs"] = origin_abs
	paths["mileage_mode"] = mileage_mode

	return paths


def plot_deviation_curves(
	tidy_df: pd.DataFrame,
	out_file: str | Path,
	mileage_pct_start: float = 0.0,
	mileage_pct_end: float = 1.0,
) -> Path:
	"""
	基于统一里程表绘制子图：实测、设计、偏差、残差(偏差-(实测-设计))。

	参数:
	- mileage_pct_start / mileage_pct_end: 里程百分比区间，范围 [0,1]。
	  例如 (0.45, 0.55) 表示只看中间 10% 里程区段。
	"""
	out_file = Path(out_file)
	out_file.parent.mkdir(parents=True, exist_ok=True)

	if "里程" not in tidy_df.columns:
		raise ValueError("tidy_df 缺少 '里程' 列，无法绘图。")

	# 指标映射：每个子图画 4 条曲线
	metrics = [
		("左高低", "实测左高低", "设计左高低", "左高低偏差"),
		("右高低", "实测右高低", "设计右高低", "右高低偏差"),
		("左轨向", "实测左轨向", "设计左轨向", "左轨向偏差"),
		("右轨向", "实测右轨向", "设计右轨向", "右轨向偏差"),
		("轨距", "实测轨距", "设计轨距", "轨距偏差"),
		("水平", "实测水平", "设计水平", "水平偏差"),
	]

	available = [m for m in metrics if all(col in tidy_df.columns for col in m[1:])]
	if not available:
		raise ValueError("未找到完整的实测/设计/偏差列组合，无法绘制子图。")

	x_all = pd.to_numeric(tidy_df["里程"], errors="coerce")
	if x_all.notna().sum() < 2:
		raise ValueError("里程列有效点不足，无法绘图。")

	# 里程百分比裁剪
	start = max(0.0, min(1.0, float(mileage_pct_start)))
	end = max(0.0, min(1.0, float(mileage_pct_end)))
	if end <= start:
		end = min(1.0, start + 0.1)

	x_min = float(x_all.min())
	x_max = float(x_all.max())
	span = x_max - x_min
	m0 = x_min + start * span
	m1 = x_min + end * span

	mask = x_all.between(m0, m1, inclusive="both")
	tdf = tidy_df.loc[mask].copy()
	x = pd.to_numeric(tdf["里程"], errors="coerce")
	if len(tdf) < 2:
		raise ValueError("选定百分比区间内数据点不足，请放宽里程区间。")

	plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
	plt.rcParams["axes.unicode_minus"] = False

	fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
	axes = axes.ravel()

	for i, (name, c_meas, c_design, c_dev) in enumerate(available[:6]):
		ax = axes[i]
		meas = pd.to_numeric(tdf[c_meas], errors="coerce")
		design = pd.to_numeric(tdf[c_design], errors="coerce")
		dev = pd.to_numeric(tdf[c_dev], errors="coerce")
		resid = dev - (meas - design)

		# 强化线型区分，便于近距离观察
		ax.plot(x, meas, linewidth=1.4, color="#1f77b4", linestyle="-", label="实测值")
		ax.plot(x, design, linewidth=1.3, color="#2ca02c", linestyle=":", label="设计值")
		ax.plot(x, dev, linewidth=1.3, color="#ff7f0e", linestyle="-.", label="偏差值")
		ax.plot(x, resid, linewidth=1.2, color="#d62728", linestyle="--", label="残差: 偏差-(实测-设计)")

		ax.set_title(name)
		ax.set_xlabel("里程")
		ax.set_ylabel("数值")
		ax.grid(True, alpha=0.35, linestyle="--")
		ax.legend(fontsize=8)

	# 若不足 6 个指标，隐藏多余子图
	for j in range(len(available[:6]), len(axes)):
		axes[j].axis("off")

	fig.suptitle(
		f"实测/设计/偏差及残差 子图（统一里程，区间 {start:.0%} - {end:.0%}）",
		fontsize=14,
	)

	fig.savefig(out_file, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return out_file


def save_table(df: pd.DataFrame, out_file: str | Path, encoding: str = "utf-8-sig") -> Path:
	"""
	保存表格文件，默认 utf-8-sig 以避免 Windows/Excel 打开中文乱码。
	如需兼容老系统，可传 encoding='gb18030'。
	"""
	out_file = Path(out_file)
	out_file.parent.mkdir(parents=True, exist_ok=True)

	def _write(target: Path):
		# CSV 使用带 BOM 的 UTF-8，Excel 双击打开中文通常不乱码
		if target.suffix.lower() == ".csv":
			df.to_csv(target, index=False, encoding=encoding)
		else:
			# 其他后缀默认也按 csv 写；你也可以后续扩展为 xlsx/parquet
			df.to_csv(target, index=False, encoding=encoding)

	try:
		_write(out_file)
	except PermissionError:
		# 文件被 Excel/编辑器占用时自动另存
		ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		fallback = out_file.with_name(f"{out_file.stem}_{ts}{out_file.suffix}")
		_write(fallback)
		out_file = fallback

	return out_file


if __name__ == "__main__":
	sample = Path(r"defect_injector\irre_file\呼局\4.16sx278-276.jdt")
	# 里程百分比绘图区间（可调）
	plot_pct_start = 0.45
	plot_pct_end = 0.55

	if sample.exists():
		data = read_jdt(sample)
		print(f"Read OK: {sample}")
		print(f"shape={data.shape}")
		print(data.head(3))

		# 输出文件名按输入 .jdt 自动命名
		in_dir = sample.parent
		in_stem = sample.stem
		out_raw_csv = in_dir / f"{in_stem}.jdt.csv"
		out_tidy_csv = in_dir / f"{in_stem}.tidy.csv"
		out_plot_png = in_dir / f"{in_stem}.deviation_subplots.png"

		out = save_table(data, out_raw_csv, encoding="utf-8-sig")
		print(f"Saved: {out} (encoding=utf-8-sig)")

		tidy = tidy_jdt_with_single_mileage(data)
		out_tidy = save_table(tidy, out_tidy_csv, encoding="utf-8-sig")
		print(f"Saved tidy: {out_tidy} (统一里程列 + Y值列)")
		print(f"tidy shape={tidy.shape}")
		print(tidy.head(3))

		# 导出为 irregularity.py 外部导入可直接读取的 4 文件格式
		ext_dir = in_dir / f"{in_stem}.external"
		ext_paths = export_for_irregularity_external_import(
			tidy,
			ext_dir,
			file_prefix=in_stem,
			use_relative_mileage=False,  # 保留绝对里程
		)
		print("Exported external files for irregularity.py:")
		print(ext_paths)

		out_fig = plot_deviation_curves(
			tidy,
			out_plot_png,
			mileage_pct_start=plot_pct_start,
			mileage_pct_end=plot_pct_end,
		)
		print(f"Saved deviation plot: {out_fig}")
	else:
		print("Sample .jdt not found.")