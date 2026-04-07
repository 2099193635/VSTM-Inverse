from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


YAML_FILES = [
    'vehicle_params.yaml',
    'rail_params.yaml',
    'fastener_kv.yaml',
    'subrail_params.yaml',
    'fastener_fdkv_params.yaml',
    'extra_force_elements.yaml',
    'antiyawer_params.yaml',
]


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'YAML 顶层必须为字典: {path}')
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='根据 sweep manifest 生成 trial 参数目录')
    parser.add_argument('--manifest', required=True, help='扫描清单 YAML 路径')
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    workspace_root = manifest_path.parent.parent.parent

    manifest = load_yaml(manifest_path)
    base_profile_dir = workspace_root / manifest.get('base_profile_dir', 'configs/standard')
    output_root = workspace_root / manifest.get('output_root', 'configs/trials/generated')
    manifest_name = manifest.get('manifest_name', manifest_path.stem)
    common = manifest.get('common', {})
    cases = manifest.get('cases', [])

    if not isinstance(cases, list) or not cases:
        raise ValueError('manifest 中 cases 必须是非空列表')

    base_yamls = {name: load_yaml(base_profile_dir / name) for name in YAML_FILES}

    generated = []
    for case in cases:
        case_id = case['case_id']
        note = case.get('note', case_id)
        updates = case.get('updates', {})
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        for yaml_name, patch in updates.items():
            merged = deepcopy(base_yamls.get(yaml_name, {}))
            deep_update(merged, patch)
            dump_yaml(case_dir / yaml_name, merged)

        meta = {
            'manifest_name': manifest_name,
            'case_id': case_id,
            'note': note,
            'base_profile_dir': str(base_profile_dir),
            'generated_profile_dir': str(case_dir),
            'common': common,
            'updated_yaml_files': list(updates.keys()),
        }
        dump_yaml(case_dir / 'case_meta.yaml', meta)
        generated.append((case_id, note, case_dir))

    print(f'已生成 {len(generated)} 组试验参数目录:')
    for case_id, note, case_dir in generated:
        print(f'- {case_id}: {note}')
        print(f'  {case_dir}')

    print('\n建议运行方式示例:')
    sample_case = generated[0][0]
    sample_dir = output_root / sample_case
    note_prefix = common.get('note_prefix', manifest_name)
    print(
        'python generate_main.py '
        f'--param_profile_dir "{sample_dir}" '
        f'--run_note "{note_prefix}_{sample_case}" '
        f'--save_dof_mode "{common.get("save_dof_mode", "vehicle")}"'
    )


if __name__ == '__main__':
    main()
