#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/workspace/VTCM_PYTHON"
SCRIPT_PATH="$ROOT_DIR/PINO/generate_dataset.py"
RUN_DIR="$ROOT_DIR/results/job_control"
LOG_DIR="$ROOT_DIR/results/job_logs"
PID_FILE="$RUN_DIR/generate_dataset.pid"
LATEST_LOG_LINK="$LOG_DIR/generate_dataset_latest.log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "未找到可用的 Python 解释器。"
  exit 1
fi

show_help() {
  cat <<'EOF'
用法:
  bash scripts/generate_dataset_job.sh start [generate_dataset.py 参数...]
  bash scripts/generate_dataset_job.sh status
  bash scripts/generate_dataset_job.sh tail
  bash scripts/generate_dataset_job.sh stop

示例:
  bash scripts/generate_dataset_job.sh start --num_train 200 --num_test 40 --hide_main_output
  bash scripts/generate_dataset_job.sh status
  bash scripts/generate_dataset_job.sh tail
  bash scripts/generate_dataset_job.sh stop

说明:
  - 使用 nohup 在服务器后台运行，即使 VS Code 或 SSH 会话断开，任务也会继续。
  - 日志目录: results/job_logs/
  - PID 文件: results/job_control/generate_dataset.pid
EOF
}

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

start_job() {
  if is_running; then
    echo "任务已在运行中，PID: $(cat "$PID_FILE")"
    echo "查看日志: $LATEST_LOG_LINK"
    exit 0
  fi

  local timestamp log_file pid
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/generate_dataset_${timestamp}.log"

  nohup "$PYTHON_BIN" "$SCRIPT_PATH" "$@" >"$log_file" 2>&1 &
  pid=$!
  echo "$pid" > "$PID_FILE"
  ln -sfn "$log_file" "$LATEST_LOG_LINK"

  echo "后台任务已启动"
  echo "PID: $pid"
  echo "日志: $log_file"
  echo "查看状态: bash scripts/generate_dataset_job.sh status"
  echo "查看日志: bash scripts/generate_dataset_job.sh tail"
}

status_job() {
  if is_running; then
    echo "任务运行中，PID: $(cat "$PID_FILE")"
    echo "最新日志: $LATEST_LOG_LINK"
  else
    echo "当前没有运行中的 generate_dataset 后台任务。"
    if [[ -f "$PID_FILE" ]]; then
      echo "检测到旧 PID 文件，正在清理。"
      rm -f "$PID_FILE"
    fi
  fi
}

tail_log() {
  if [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]]; then
    tail -f "$LATEST_LOG_LINK"
  else
    echo "尚未找到日志文件，请先执行 start。"
    exit 1
  fi
}

stop_job() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid"
    rm -f "$PID_FILE"
    echo "已停止后台任务，PID: $pid"
  else
    echo "当前没有运行中的后台任务。"
    rm -f "$PID_FILE"
  fi
}

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$ACTION" in
  start)
    start_job "$@"
    ;;
  status)
    status_job
    ;;
  tail)
    tail_log
    ;;
  stop)
    stop_job
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    echo "未知命令: $ACTION"
    echo
    show_help
    exit 1
    ;;
esac