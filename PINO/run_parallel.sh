#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/VTCM_PYTHON"
SCRIPT="PINO/VTCM_physicis_informed_fno.py"

EXP0="${1:-exp_gpu0}"
EXP1="${2:-exp_gpu1}"
MEMORY_THRESHOLD_MIB="${MEMORY_THRESHOLD_MIB:-2048}"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/outputs_pino"

mapfile -t GPU_STATUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

AVAILABLE_GPUS=()
for row in "${GPU_STATUS[@]}"; do
  gpu_index="$(echo "$row" | cut -d',' -f1 | xargs)"
  memory_used="$(echo "$row" | cut -d',' -f2 | xargs)"
  if [[ "$memory_used" =~ ^[0-9]+$ ]] && (( memory_used < MEMORY_THRESHOLD_MIB )); then
    AVAILABLE_GPUS+=("$gpu_index")
  fi
done

if (( ${#AVAILABLE_GPUS[@]} == 0 )); then
  echo "No idle GPU found."
  echo "Current GPU memory usage:"
  printf '  %s\n' "${GPU_STATUS[@]}"
  echo "Tip: raise MEMORY_THRESHOLD_MIB if you want to treat lightly used GPUs as available."
  exit 1
fi

EXPERIMENTS=("$EXP0" "$EXP1")
LAUNCHED=()

for i in "${!AVAILABLE_GPUS[@]}"; do
  if (( i >= ${#EXPERIMENTS[@]} )); then
    break
  fi

  gpu="${AVAILABLE_GPUS[$i]}"
  exp="${EXPERIMENTS[$i]}"
  log_file="$PROJECT_ROOT/outputs_pino/${exp}.launcher.log"

  CUDA_VISIBLE_DEVICES="$gpu" nohup /usr/bin/python "$SCRIPT" experiment_name="$exp" \
    > "$log_file" 2>&1 &
  pid=$!
  LAUNCHED+=("GPU $gpu -> PID $pid, experiment_name=$exp, log=outputs_pino/${exp}.launcher.log")
done

if (( ${#LAUNCHED[@]} == 0 )); then
  echo "No job launched."
  exit 1
fi

echo "Detected GPU memory usage (MiB):"
printf '  %s\n' "${GPU_STATUS[@]}"
echo
echo "Launched training jobs:"
printf '  %s\n' "${LAUNCHED[@]}"
echo
echo "Useful commands:"
echo "  ps -ef | grep '$SCRIPT' | grep -v grep"
echo "  nvidia-smi"
echo "  tail -f outputs_pino/<experiment_name>.launcher.log"
echo
echo "Env override: MEMORY_THRESHOLD_MIB=$MEMORY_THRESHOLD_MIB"
