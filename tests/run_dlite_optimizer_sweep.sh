#!/usr/bin/env bash
# Sequential remote experiments. All parameter changes are CLI overrides.
# Set the same SWEEP_ID on every node of a multi-node job.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PREVIEW=0
if [[ "${1:-}" == --dry-run ]]; then
  PREVIEW=1
  shift
fi
CONFIG="${1:-scripts/config/dlite_sft.yaml}"
SWEEP_ID="${SWEEP_ID:-muon_adamw_sweep}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-dlite_sft_qwen3_4b}"
KEEPER_DIR="${KEEPER_DIR:-$(dirname "$PROJECT_DIR")}"
LOG_DIR="whz_mtp_logs/${SWEEP_ID}/$(hostname)_rank${PET_NODE_RANK:-${NODE_RANK:-0}}"
OUTPUT_ROOT="outputs/${SWEEP_ID}"
export WANDB_MODE=offline

if (( PREVIEW )); then
  export DRY_RUN=1
else
  unset DRY_RUN
  [[ -f "$CONFIG" ]]
  [[ -x "$KEEPER_DIR/stop_keeper.sh" && -x "$KEEPER_DIR/start_keeper.sh" ]]
  command -v setsid >/dev/null
  mkdir -p "$(dirname "$LOG_DIR")"
  # Refuse to overwrite a previous sweep's logs. Choose a new SWEEP_ID to rerun.
  mkdir "$LOG_DIR"
  # Preserve the baseline across all five runs without editing any YAML values.
  cp "$CONFIG" "$LOG_DIR/baseline.yaml"
  CONFIG="$LOG_DIR/baseline.yaml"
fi

TRAIN_PID=""
cleanup() {
  local status=$?
  trap - EXIT
  echo "实验队列结束，退出码：$status；启动 keeper"
  if "$KEEPER_DIR/start_keeper.sh"; then
    :
  elif (( status == 0 )); then
    status=1
  fi
  exit "$status"
}
interrupt_training() {
  local status="$1"
  trap '' INT TERM
  if [[ -n "$TRAIN_PID" ]]; then
    kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
    TRAIN_PID=""
  fi
  exit "$status"
}

if (( ! PREVIEW )); then
  trap cleanup EXIT
  trap 'interrupt_training 130' INT
  trap 'interrupt_training 143' TERM
  "$KEEPER_DIR/stop_keeper.sh"
fi

wandb_name_from_overrides() {
  local result="$WANDB_NAME_PREFIX" option short value
  while (( $# )); do
    option="$1"
    shift
    case "$option" in
      --use-muon) result+="_um1"; continue ;;
      --no-use-muon) result+="_um0"; continue ;;
      --muon-lr) short=mlr ;;
      --muon-momentum) short=mm ;;
      --muon-weight-decay) short=mwd ;;
      --muon-adamw-lr) short=malr ;;
      --muon-adamw-weight-decay) short=mawd ;;
      --muon-matrices-per-rank) short=mpr ;;
      --learning-rate) short=lr ;;
      --weight-decay) short=wd ;;
      *) short="${option#--}" ;;
    esac
    if (( $# == 0 )); then
      echo "Missing value for experiment override: $option" >&2
      return 2
    fi
    value="$1"
    shift
    result+="_${short}${value}"
  done
  printf '%s\n' "$result"
}

run_experiment() {
  local name="$1"
  shift
  local wandb_name
  wandb_name="$(wandb_name_from_overrides "$@")"
  local command=(
    bash scripts/dlite/run_training_dlite_sft.sh "$CONFIG"
    "$@"
    --output-dir "$OUTPUT_ROOT/$name"
    --wandb-name "$wandb_name"
  )
  echo "启动实验：$name"
  if (( PREVIEW )); then
    "${command[@]}"
    return
  fi
  local logfile="$LOG_DIR/$name.log"
  printf '%q ' "${command[@]}" > "$logfile"
  printf '\n' >> "$logfile"
  setsid "${command[@]}" >> "$logfile" 2>&1 &
  TRAIN_PID=$!
  local status=0
  wait "$TRAIN_PID" || status=$?
  TRAIN_PID=""
  printf '%s\t%s\n' "$name" "$status" >> "$LOG_DIR/status.tsv"
  echo "实验结束：$name，退出码：$status，日志：$logfile"
  # Fail fast: do not silently run subsequent distributed experiments after failure.
  return "$status"
}

run_experiment exp1_muon_lr001 --use-muon --muon-lr 0.01
run_experiment exp2_muon_lr005 --use-muon --muon-lr 0.05
run_experiment exp3_adamw_wd0 --no-use-muon --weight-decay 0.0
run_experiment exp4_adamw_wd01 --no-use-muon --weight-decay 0.1
run_experiment exp5_muon_wd001 --use-muon --muon-weight-decay 0.01 --muon-adamw-weight-decay 0.01
