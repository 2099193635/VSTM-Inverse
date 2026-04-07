from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

INTERNAL_COMMON_KEYS = {'note_prefix', 'description'}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'未找到文件: {path}')
    with path.open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f'YAML 顶层必须为字典: {path}')
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def normalize_cases(cases: list[dict[str, Any]], include: list[str] | None, exclude: list[str] | None) -> list[dict[str, Any]]:
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    filtered: list[dict[str, Any]] = []
    for case in cases:
        case_id = case.get('case_id')
        if not case_id:
            raise ValueError('manifest 中每个 case 都必须包含 case_id')
        if include_set and case_id not in include_set:
            continue
        if case_id in exclude_set:
            continue
        filtered.append(case)
    return filtered


def append_cli_arg(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        value = 'On' if value else 'Off'
    if isinstance(value, (dict, list, tuple)):
        raise ValueError(f'common 参数暂不支持复合类型: {key}')
    command.extend([f'--{key}', str(value)])


def build_case_command(
    python_exe: str,
    workspace_root: Path,
    common: dict[str, Any],
    profile_dir: Path,
    case_id: str,
    manifest_name: str,
    extra_args: list[str],
) -> list[str]:
    command = [python_exe, str(workspace_root / 'generate_main.py')]

    for key, value in common.items():
        if key in INTERNAL_COMMON_KEYS:
            continue
        append_cli_arg(command, key, value)

    note_prefix = str(common.get('note_prefix', 'sweep'))
    run_note = f'{note_prefix}_{case_id}'
    if 'project_name' not in common:
        command.extend(['--project_name', manifest_name])
    command.extend(['--param_profile_dir', str(profile_dir)])
    command.extend(['--run_note', run_note])
    command.extend(extra_args)
    return command


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='按 sweep manifest 批量运行 generate_main.py')
    parser.add_argument('--manifest', required=True, help='扫描清单 YAML 路径')
    parser.add_argument('--python-exe', default=sys.executable, help='运行 generate_main.py 的 Python 可执行文件')
    parser.add_argument('--build-first', action='store_true', help='运行前先调用 build_param_sweep.py 生成/更新 trial 参数目录')
    parser.add_argument('--dry-run', action='store_true', help='只打印将执行的命令，不实际运行')
    parser.add_argument('--cases', nargs='*', help='只运行指定 case_id 列表')
    parser.add_argument('--skip-cases', nargs='*', help='跳过指定 case_id 列表')
    parser.add_argument('--stop-on-error', action='store_true', help='遇到首个失败 case 即停止')
    parser.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[], help='透传给 generate_main.py 的额外参数（放在命令最后）')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    manifest_path = Path(args.manifest).resolve()
    workspace_root = manifest_path.parent.parent.parent
    manifest = load_yaml(manifest_path)
    manifest_name = str(manifest.get('manifest_name', manifest_path.stem))

    output_root = workspace_root / manifest.get('output_root', 'configs/trials/generated')
    common = manifest.get('common', {})
    if not isinstance(common, dict):
        raise ValueError('manifest.common 必须为字典')
    cases = manifest.get('cases', [])
    if not isinstance(cases, list) or not cases:
        raise ValueError('manifest 中 cases 必须是非空列表')

    selected_cases = normalize_cases(cases, args.cases, args.skip_cases)
    if not selected_cases:
        raise ValueError('筛选后没有可运行的 case')

    if args.build_first:
        build_command = [
            args.python_exe,
            str(workspace_root / 'utils' / 'build_param_sweep.py'),
            '--manifest',
            str(manifest_path),
        ]
        print('[build] ' + subprocess.list2cmdline(build_command))
        if not args.dry_run:
            subprocess.run(build_command, cwd=workspace_root, check=True)

    run_started_at = datetime.now()
    log_entries: list[dict[str, Any]] = []

    print(f'将运行 {len(selected_cases)} 组 case，输出目录根路径: {output_root}')
    for index, case in enumerate(selected_cases, start=1):
        case_id = str(case['case_id'])
        profile_dir = output_root / case_id
        if not profile_dir.exists():
            raise FileNotFoundError(f'未找到 case 参数目录: {profile_dir}，可先加 --build-first')

        command = build_case_command(
            python_exe=args.python_exe,
            workspace_root=workspace_root,
            common=common,
            profile_dir=profile_dir,
            case_id=case_id,
            manifest_name=manifest_name,
            extra_args=args.extra_args,
        )
        command_str = subprocess.list2cmdline(command)
        print(f'[{index}/{len(selected_cases)}] {case_id}')
        print('  ' + command_str)

        case_started = time.perf_counter()
        status = 'dry-run'
        return_code = None
        if not args.dry_run:
            completed = subprocess.run(command, cwd=workspace_root)
            return_code = completed.returncode
            status = 'success' if completed.returncode == 0 else 'failed'
        elapsed = round(time.perf_counter() - case_started, 3)
        print(f'  -> {status}, elapsed={elapsed:.3f}s')

        log_entries.append({
            'case_id': case_id,
            'profile_dir': str(profile_dir),
            'command': command,
            'status': status,
            'return_code': return_code,
            'elapsed_s': elapsed,
        })

        if status == 'failed' and args.stop_on_error:
            break

    summary = {
        'manifest': str(manifest_path),
        'output_root': str(output_root),
        'project_name': common.get('project_name', manifest_name),
        'run_started_at': run_started_at.isoformat(timespec='seconds'),
        'run_finished_at': datetime.now().isoformat(timespec='seconds'),
        'dry_run': args.dry_run,
        'build_first': args.build_first,
        'total_cases': len(selected_cases),
        'success_cases': sum(1 for item in log_entries if item['status'] == 'success'),
        'failed_cases': sum(1 for item in log_entries if item['status'] == 'failed'),
        'entries': log_entries,
    }
    log_name = f"sweep_run_{run_started_at.strftime('%Y%m%d_%H%M%S')}.yaml"
    log_path = output_root / log_name
    dump_yaml(log_path, summary)
    print(f'\n批量运行日志已写入: {log_path}')


if __name__ == '__main__':
    main()
