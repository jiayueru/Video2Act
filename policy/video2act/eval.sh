#!/bin/bash
# Video2Act eval entry for RoboTwin 2.0.
#
# Usage:
#   bash policy/video2act/eval.sh <task_name> <task_config> <ckpt_dir> <checkpoint_id> <seed> <gpu_id>
#
# <ckpt_dir>: full path to a checkpoint-<step> directory (the directory that
#             contains pytorch_model.bin or pytorch_model/mp_rank_00_model_states.pt).
#             Pass the absolute path — the loader walks the directory directly.
# <checkpoint_id>: a small integer used purely as a label in eval_result paths
#                  (does NOT have to equal the training step).
#
# Required env (export before launch — see policy/video2act/run_scripts/README.md):
#   VIDEO2ACT_WAN_DIT_PATH       Wan2.2-5B-Robot/checkpoint.safetensors
#   VIDEO2ACT_WAN_VAE_PATH       wan-t2v-5b/Wan2.2_VAE.pth
#   VIDEO2ACT_WAN_TEXT_ENCODER_PATH   wan-t2v-5b/models_t5_umt5-xxl-enc-bf16.pth
#   VIDEO2ACT_WAN_TOKENIZER_PATH umt5-xxl/   (UMT5 tokenizer dir)
#
# Optional env:
#   MODEL_CFG  path to model_config/*.yml (default: model_config/video2act_example.yml)
#   PROCESSED_DATA_ROOT  used to resolve fixed lang/wan text embeddings per task
#                        (default: $REPO_ROOT/policy/video2act/processed_data)
#   FIXED_LANG_EMBED  True | False (default True if both embed files exist; else False)
#   POLICY_CONDA_ENV  conda env name to activate (default RoboTwin)
#   CACHE_RATIO       Slow-fast WAN cache ratio override (1=off, N>1 = 1:N).
#                     Maps to VIDEO2ACT_CACHE_RATIO; overrides yaml enable_cache /
#                     cache_ratio fields. Common values: 2, 4, 8, 16.

set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <task_name> <task_config> <ckpt_dir> <checkpoint_id> <seed> <gpu_id>" >&2
    exit 2
fi

task_name="$1"
task_config="$2"
ckpt_dir="$3"
checkpoint_id="$4"
seed="$5"
gpu_id="$6"

policy_name="${POLICY_NAME:-video2act}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_CFG="${MODEL_CFG:-${SCRIPT_DIR}/model_config/video2act_example.yml}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${SCRIPT_DIR}/processed_data}"
POLICY_CONDA_ENV="${POLICY_CONDA_ENV:-RoboTwin}"

# --- Pre-flight checks ---
for v in VIDEO2ACT_WAN_DIT_PATH VIDEO2ACT_WAN_VAE_PATH \
         VIDEO2ACT_WAN_TEXT_ENCODER_PATH VIDEO2ACT_WAN_TOKENIZER_PATH \
         VISION_ENCODER_NAME TEXT_ENCODER_NAME; do
    if [ -z "${!v:-}" ] || [ ! -e "${!v}" ]; then
        echo "ERROR: env $v not set or path missing: ${!v:-<unset>}" >&2
        echo "       See the env-export block in the top-level README." >&2
        exit 1
    fi
done
if [ ! -d "$ckpt_dir" ]; then
    echo "ERROR: ckpt_dir does not exist: $ckpt_dir" >&2
    exit 1
fi
if [ ! -f "$ckpt_dir/pytorch_model.bin" ] && \
   [ ! -f "$ckpt_dir/pytorch_model/mp_rank_00_model_states.pt" ]; then
    echo "ERROR: ckpt_dir contains neither pytorch_model.bin nor pytorch_model/mp_rank_00_model_states.pt: $ckpt_dir" >&2
    exit 1
fi
if [ ! -f "$MODEL_CFG" ]; then
    echo "ERROR: model_config not found: $MODEL_CFG" >&2
    exit 1
fi

# --- Per-task fixed embeddings (auto-detect) ---
task_data_dir="${PROCESSED_DATA_ROOT}/${task_name}-aloha-agilex_clean_50-50_wan22_832x480"
lang_embed="${task_data_dir}/episode_0/instructions/lang_embed_0.pt"
wan_text="${task_data_dir}/text_embeddings/seen_0000.pkl"

fixed_lang_embed="${FIXED_LANG_EMBED:-}"
if [ -z "$fixed_lang_embed" ]; then
    if [ -f "$lang_embed" ] && [ -f "$wan_text" ]; then
        fixed_lang_embed=True
    else
        fixed_lang_embed=False
    fi
fi

# --- env ---
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" = "$POLICY_CONDA_ENV" ]; then
    : # already in target env
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$POLICY_CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES="$gpu_id"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/video2act_triton_${USER:-root}}"
mkdir -p "$TRITON_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/policy/video2act:${PYTHONPATH:-}"

# Slow-fast WAN cache ratio override (optional). When set, overrides the
# enable_cache / cache_ratio fields in the architecture YAML.
#   CACHE_RATIO=1   -> cache disabled (recompute every step)
#   CACHE_RATIO=N>1 -> recompute WAN once every N policy calls
if [ -n "${CACHE_RATIO:-}" ]; then
    export VIDEO2ACT_CACHE_RATIO="$CACHE_RATIO"
fi

# Optional headless EGL setup (RoboTwin 2.0 helper, if present)
[ -f "${REPO_ROOT}/eval_script/setup_render_env.sh" ] && \
    source "${REPO_ROOT}/eval_script/setup_render_env.sh"

echo -e "\033[33mtask=${task_name}/${task_config}  seed=${seed}  gpu=${gpu_id}\033[0m"
echo "ckpt_dir          : $ckpt_dir"
echo "model_config_path : $MODEL_CFG"
echo "fixed_lang_embed  : $fixed_lang_embed"
[ -n "${VIDEO2ACT_CACHE_RATIO:-}" ] && \
    echo "cache_ratio       : 1:${VIDEO2ACT_CACHE_RATIO} (env override)"
[ "$fixed_lang_embed" = "True" ] && {
    echo "  lang_embed      : $lang_embed"
    echo "  wan_text_embed  : $wan_text"
}

cd "$REPO_ROOT"

cmd=(
    python script/eval_policy.py
    --config "policy/${policy_name}/deploy_policy.yml"
    --overrides
    --task_name "$task_name"
    --task_config "$task_config"
    --ckpt_setting "$ckpt_dir"
    --seed "$seed"
    --checkpoint_id "$checkpoint_id"
    --policy_name "$policy_name"
    --model_config_path "$MODEL_CFG"
    --fixed_lang_embed "$fixed_lang_embed"
)
if [ "$fixed_lang_embed" = "True" ]; then
    cmd+=(--fixed_lang_embed_path "$lang_embed"
          --fixed_wan_text_embed_path "$wan_text")
fi

PYTHONWARNINGS=ignore::UserWarning "${cmd[@]}"
