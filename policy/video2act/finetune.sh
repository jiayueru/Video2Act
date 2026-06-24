#!/bin/bash
# Multi-GPU shared-queue finetune for video2act.
#
# Each GPU = 1 worker; pulls a CONFIG_NAME from the queue and runs single-card
# `accelerate launch main_video2act.py` on that card. Failed configs are
# logged and the worker moves on to the next one.
#
# Usage:
#   bash finetune.sh <single_cfg>                       # backward-compat one-shot
#   CONFIGS="cfg_a cfg_b cfg_c" bash finetune.sh        # multi-config queue
#
# Common overrides:
#   CONFIGS="cfg_a cfg_b cfg_c" \
#   GPUS_LIST="0 1 2 3 4 5 6 7" \
#   FORCE_RERUN=0 DRY_RUN=0 BASE_PORT=28499 \
#     bash finetune.sh
#
# Note: the YAML field `cuda_visible_device` is IGNORED; GPU placement comes
#       from $GPUS_LIST. Each run is single-card, so effective batch size =
#       train_batch_size * 1 (the original script multiplied by all GPUs in
#       CUDA_USE) -- you may want to retune lr / grad_accum.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- env defaults ----
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-google/t5-v1_1-xxl}"
export VISION_ENCODER_NAME="${VISION_ENCODER_NAME:-../weights/RDT/siglip-so400m-patch14-384}"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export WANDB_PROJECT="${WANDB_PROJECT:-video2act}"
export WANDB_MODE="${WANDB_MODE:-offline}"
REPORT_TO="${REPORT_TO:-wandb}"

# Triton cache: default is /root/.triton, which is read-only on many shared
# environments. Point it at a writable per-user temp dir.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/video2act_triton_$USER}"
mkdir -p "$TRITON_CACHE_DIR"

# DeepSpeed: avoid runtime CUDA op compilation surprises.
export DS_ACCELERATOR="${DS_ACCELERATOR:-cuda}"

# flash_attn ABI: ensure libstdc++ from the active conda env is preferred over
# the system one (CXXABI_1.3.15 mismatches otherwise). CONDA_PREFIX is set when
# the user has activated their training env; we prepend its lib dir if present.
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# ---- queue inputs ----
CONFIGS="${CONFIGS:-${1:-}}"
if [ -z "$CONFIGS" ]; then
  echo "ERROR: no CONFIGS specified. Pass via env: CONFIGS=\"cfg_a cfg_b\" or as the first positional arg." >&2
  exit 1
fi
read -ra CONFIG_LIST <<< "$CONFIGS"

GPUS_LIST="${GPUS_LIST:-0 1 2 3 4 5 6 7}"
read -ra GPUS <<< "$GPUS_LIST"
[ "${#GPUS[@]}" -ge 1 ] || { echo "ERROR: GPUS_LIST is empty." >&2; exit 1; }

FORCE_RERUN="${FORCE_RERUN:-0}"
DRY_RUN="${DRY_RUN:-0}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-./configs/zero2.json}"
BASE_PORT="${BASE_PORT:-28499}"

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/finetune_$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOG_DIR/master.log"
mkdir -p "$LOG_DIR"

# ---- sanity checks ----
[ -f "$SCRIPT_DIR/main_video2act.py" ] || { echo "ERROR: main_video2act.py not found at $SCRIPT_DIR" >&2; exit 1; }
[ -f "$DEEPSPEED_CFG" ] || { echo "ERROR: deepspeed config not found: $DEEPSPEED_CFG" >&2; exit 1; }
for cfg in "${CONFIG_LIST[@]}"; do
  yml="model_config/$cfg.yml"
  [ -f "$yml" ] || { echo "ERROR: config file $yml does not exist" >&2; exit 1; }
done

read_yaml_field() {
  python scripts/read_yaml.py "$1" "$2" | tr -d '"'
}

has_finished() {
  local cfg=$1
  [ "$FORCE_RERUN" = "1" ] && return 1
  local out_dir max_steps
  out_dir=$(read_yaml_field "model_config/$cfg.yml" checkpoint_path)
  max_steps=$(read_yaml_field "model_config/$cfg.yml" max_train_steps)
  [ -d "$out_dir/checkpoint-$max_steps" ]
}

claim_task() {
  local idx_file="$LOG_DIR/next_idx"
  local lock_dir="$LOG_DIR/queue.lock"
  local idx cfg

  while ! mkdir "$lock_dir" 2>/dev/null; do
    sleep 0.2
  done

  idx="$(cat "$idx_file")"
  if [ "$idx" -ge "${#CONFIG_LIST[@]}" ]; then
    rmdir "$lock_dir"
    return 1
  fi

  cfg="${CONFIG_LIST[$idx]}"
  echo $((idx + 1)) > "$idx_file"
  rmdir "$lock_dir"
  printf '%s\n' "$cfg"
}

run_one_config() {
  local gpu=$1 cfg=$2 gpu_log=$3
  local yml="model_config/$cfg.yml"
  local port=$((BASE_PORT + gpu))

  if has_finished "$cfg"; then
    echo "[$(date '+%F %T')] [GPU $gpu] SKIP  $cfg (final checkpoint exists)" | tee -a "$MASTER_LOG" "$gpu_log"
    return 0
  fi

  local pretrained config_path train_bs sample_bs max_steps ckpt_period sample_period
  local ckpt_total_limit lr_sched lr dl_workers state_noise grad_accum out_dir
  pretrained=$(read_yaml_field "$yml" pretrained_model_name_or_path)
  config_path=$(read_yaml_field "$yml" config_path)
  train_bs=$(read_yaml_field "$yml" train_batch_size)
  sample_bs=$(read_yaml_field "$yml" sample_batch_size)
  max_steps=$(read_yaml_field "$yml" max_train_steps)
  ckpt_period=$(read_yaml_field "$yml" checkpointing_period)
  sample_period=$(read_yaml_field "$yml" sample_period)
  ckpt_total_limit=$(read_yaml_field "$yml" checkpoints_total_limit)
  lr_sched=$(read_yaml_field "$yml" lr_scheduler)
  lr=$(read_yaml_field "$yml" learning_rate)
  dl_workers=$(read_yaml_field "$yml" dataloader_num_workers)
  state_noise=$(read_yaml_field "$yml" state_noise_snr)
  grad_accum=$(read_yaml_field "$yml" gradient_accumulation_steps)
  out_dir=$(read_yaml_field "$yml" checkpoint_path)
  mkdir -p "$out_dir"

  echo "[$(date '+%F %T')] [GPU $gpu] START $cfg  (out=$out_dir port=$port)" | tee -a "$MASTER_LOG" "$gpu_log"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[$(date '+%F %T')] [GPU $gpu] DRY_RUN $cfg" | tee -a "$MASTER_LOG" "$gpu_log"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -m data.compute_dataset_stat_hdf5 --task_name "$cfg" >> "$gpu_log" 2>&1

  if CUDA_VISIBLE_DEVICES="$gpu" WANDB_DEFAULT_RUN_NAME="$cfg" \
    accelerate launch --num_processes=1 --main_process_port="$port" main_video2act.py \
      --deepspeed="$DEEPSPEED_CFG" \
      --config_path="$config_path" \
      --pretrained_model_name_or_path="$pretrained" \
      --pretrained_text_encoder_name_or_path="$TEXT_ENCODER_NAME" \
      --pretrained_vision_encoder_name_or_path="$VISION_ENCODER_NAME" \
      --output_dir="$out_dir" \
      --train_batch_size="$train_bs" \
      --sample_batch_size="$sample_bs" \
      --max_train_steps="$max_steps" \
      --checkpointing_period="$ckpt_period" \
      --sample_period="$sample_period" \
      --checkpoints_total_limit="$ckpt_total_limit" \
      --lr_scheduler="constant" \
      --learning_rate="$lr" \
      --mixed_precision="bf16" \
      --dataloader_num_workers="$dl_workers" \
      --image_aug \
      --dataset_type="finetune" \
      --state_noise_snr="$state_noise" \
      --load_from_hdf5 \
      --report_to="$REPORT_TO" \
      --precomp_lang_embed \
      --gradient_accumulation_steps="$grad_accum" \
      --model_config_path="$yml" \
      --CONFIG_NAME="$cfg" \
      >> "$gpu_log" 2>&1; then
    echo "[$(date '+%F %T')] [GPU $gpu] DONE  $cfg" | tee -a "$MASTER_LOG" "$gpu_log"
  else
    local st=$?
    echo "[$(date '+%F %T')] [GPU $gpu] FAILED $cfg exit=$st; continuing" | tee -a "$MASTER_LOG" "$gpu_log"
  fi
}

run_worker() {
  local gpu=$1
  local gpu_log="$LOG_DIR/gpu${gpu}.log"
  local cfg

  while cfg="$(claim_task)"; do
    run_one_config "$gpu" "$cfg" "$gpu_log"
  done
  echo "[$(date '+%F %T')] [GPU $gpu] idle; queue empty" | tee -a "$MASTER_LOG" "$gpu_log"
}

# ---- summary header ----
{
  echo "===== video2act finetune queue | $(date) ====="
  echo "SCRIPT_DIR    : $SCRIPT_DIR"
  echo "GPUs          : ${GPUS[*]}"
  echo "Configs (#${#CONFIG_LIST[@]}):"
  printf '  %s\n' "${CONFIG_LIST[@]}"
  echo "DEEPSPEED_CFG : $DEEPSPEED_CFG"
  echo "BASE_PORT     : $BASE_PORT  (each worker uses BASE_PORT + gpu_id)"
  echo "FORCE_RERUN   : $FORCE_RERUN"
  echo "DRY_RUN       : $DRY_RUN"
  echo "REPORT_TO     : $REPORT_TO  (WANDB_PROJECT=$WANDB_PROJECT, WANDB_MODE=$WANDB_MODE)"
  echo "LOG_DIR       : $LOG_DIR"
} | tee "$MASTER_LOG"

echo 0 > "$LOG_DIR/next_idx"
for gpu in "${GPUS[@]}"; do
  run_worker "$gpu" &
done
wait

echo "===== All done: $(date) =====" | tee -a "$MASTER_LOG"
